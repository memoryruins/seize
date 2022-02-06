use crate::{resize, Pinned};

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::Range;
use std::ptr;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicPtr, AtomicUsize};
use std::sync::Arc;

use lock_api::RawMutex;
use parking_lot::Mutex;
use seize::*;

pub struct HashMap<K, V, S = RandomState> {
    // The maximum number of elements per lock before we re-allocate.
    budget: AtomicUsize,
    // The builder used to hash map keys.
    hash_builder: S,
    // The internal state of this map.
    //
    // Wrapping this in a separate struct allows us
    // to atomically swap everything at once.
    table: AtomicPtr<Linked<Table<K, V>>>,
    // Instead of adding garbage to the default global collector
    // we add it to a local collector tied to this particular map.
    //
    // Since the global collector might destroy garbage arbitrarily
    // late in the future, we would have to add `K: 'static` and `V: 'static`
    // bounds. But a local collector will destroy all remaining garbage when
    // the map is dropped, so we can accept non-'static keys and values.
    collector: Collector,
    // The lock acquired when resizing.
    resize: Arc<Mutex<()>>,
}

struct Table<K, V> {
    // The hashtable.
    buckets: Box<[AtomicPtr<Linked<Node<K, V>>>]>,
    // A set of locks, each guarding a number of buckets.
    //
    // Locks are shared across table instances during
    // resizing, hence the `Arc`s.
    locks: Arc<[Arc<Mutex<()>>]>,
    // The number of elements guarded by each lock.
    counts: Box<[AtomicUsize]>,
}

impl<K, V> Table<K, V> {
    fn new(buckets_len: usize, locks_len: usize) -> Self {
        Self {
            buckets: std::iter::repeat_with(AtomicPtr::default)
                .take(buckets_len)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            locks: std::iter::repeat_with(Default::default)
                .take(locks_len)
                .collect::<Vec<_>>()
                .into(),
            counts: std::iter::repeat_with(Default::default)
                .take(locks_len)
                .collect::<Vec<_>>()
                .into(),
        }
    }

    fn len(&self) -> usize {
        // TODO: is this too slow? should we store a cached total length?
        self.counts
            .iter()
            .fold(0, |acc, c| acc + c.load(Ordering::Relaxed))
    }

    /// Acquires a contiguous range of locks for this hash bucket.
    fn lock_range(&self, range: Range<usize>) -> impl Drop + '_ {
        for i in range.clone() {
            unsafe {
                self.locks[i].raw().lock();
            }
        }

        /// Unlocks the specified locks on drop.
        struct Guard<'a, K, V> {
            table: &'a Table<K, V>,
            range: Range<usize>,
        }

        impl<K, V> Drop for Guard<'_, K, V> {
            fn drop(&mut self) {
                for i in self.range.clone() {
                    unsafe {
                        self.table.locks[i].raw().unlock();
                    }
                }
            }
        }

        Guard { table: self, range }
    }
}

//A singly-linked list representing a bucket in the hashtable
struct Node<K, V> {
    // The key of this node.
    key: K,
    // The value of this node, heap-allocated in order to be shared
    // across buckets instances during resize operations.
    value: NonNull<V>,
    // The next node in the linked-list.
    next: AtomicPtr<Linked<Self>>,
    // The hashcode of `key`.
    hash: u64,
}

impl<K, V> HashMap<K, V, RandomState> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use seize::HashMap;
    /// let map: HashMap<&str, i32> = HashMap::new();
    /// ```
    pub fn new() -> Self {
        Self::with_hasher(RandomState::new())
    }

    /// Creates an empty `HashMap` with the specified capacity.
    ///
    ///
    /// Setting an initial capacity is a best practice if you know a size estimate
    /// up front. However, there is no guarantee that the `HashMap` will not resize
    /// if `capacity` elements are inserted. The map will resize based on key collision,
    /// so bad key distribution may cause a resize before `capacity` is reached.
    ///
    /// # Examples
    ///
    /// ```
    /// use seize::HashMap;
    /// let map: HashMap<&str, i32> = HashMap::with_capacity(10);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> HashMap<K, V, S> {
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The created map has the default initial capacity.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use seize::HashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let map = HashMap::with_hasher(s);
    /// let pinned = map.pin();
    ///
    /// pinned.insert(1, 2);
    /// ```
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            resize: Arc::new(Mutex::new(())),
            budget: AtomicUsize::new(0),
            table: AtomicPtr::default(),
            collector: Collector::new(),
        }
    }

    pub fn with_capacity_hasher_and_collector(
        capacity: usize,
        hash_builder: S,
        collector: Collector,
    ) -> HashMap<K, V, S> {
        if capacity == 0 {
            return Self::with_hasher(hash_builder);
        }

        let lock_count = resize::initial_locks(capacity);

        Self {
            resize: Arc::new(Mutex::new(())),
            budget: AtomicUsize::new(resize::budget(capacity, lock_count)),
            table: AtomicPtr::new(collector.link_boxed(Table::new(capacity, lock_count))),
            collector,
            hash_builder,
        }
    }

    /// Creates an empty `HashMap` with the specified capacity, using `hash_builder`
    /// to hash the keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use seize::HashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = HashMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, 2);
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<K, V, S> {
        Self::with_capacity_hasher_and_collector(capacity, hash_builder, Collector::new())
    }
}

impl<K, V, S> HashMap<K, V, S> {
    /// Returns a reference to the map pinned to the current thread.
    ///
    /// The only way to access a map is through a pinned reference, which,
    /// when dropped, will trigger garbage collection.
    pub fn pin(&self) -> Pinned<'_, K, V, S> {
        Pinned {
            map: self,
            guard: self.collector.enter(),
        }
    }

    /// Returns the number of elements in the map.
    pub(crate) fn len(&self, guard: &Guard<'_>) -> usize {
        let table = unsafe { guard.protect(&self.table).as_ref() };

        match table {
            Some(table) => table.len(),
            None => 0,
        }
    }

    /// Returns `true` if the map contains no elements.
    pub(crate) fn is_empty(&self, guard: &Guard<'_>) -> bool {
        self.len(guard) == 0
    }

    // /// An iterator visiting all key-value pairs in arbitrary order.
    // pub(crate) fn iter<'a>(&'a self, guard: &'a Guard<'a, Protection>) -> Iter<'a, K, V> {
    //     let table = guard.protect(|| self.table.load(Ordering::Acquire));
    //     let first_node = unsafe { table.as_ref().and_then(|t| t.buckets.get(0)) };

    //     Iter {
    //         size: self.len(guard),
    //         current_node: first_node,
    //         table,
    //         current_bucket: 0,
    //         guard,
    //     }
    // }

    // /// An iterator visiting all keys in arbitrary order.
    // pub(crate) fn keys<'a>(&'a self, guard: &'a Guard<'a, Protection>) -> Keys<'a, K, V> {
    //     Keys {
    //         iter: self.iter(guard),
    //     }
    // }

    // /// An iterator visiting all values in arbitrary order.
    // pub(crate) fn values<'a>(&'a self, guard: &'a Guard<'a, Protection>) -> Values<'a, K, V> {
    //     Values {
    //         iter: self.iter(guard),
    //     }
    // }

    fn init_table<'a>(
        &'a self,
        capacity: Option<usize>,
        guard: &'a Guard<'a>,
    ) -> &'a Linked<Table<K, V>> {
        let _guard = self.resize.lock();

        let capacity = capacity.unwrap_or(resize::DEFAULT_BUCKETS);
        match unsafe { guard.protect(&self.table).as_ref() } {
            Some(table) => table,
            None => {
                let new_table = self
                    .collector
                    .link_boxed(Table::new(capacity, resize::initial_locks(capacity)));

                self.table.store(new_table, Ordering::Release);
                unsafe { &*new_table }
            }
        }
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Returns a reference to the value corresponding to the key.
    pub(crate) fn get<'a, Q: ?Sized>(&'a self, key: &Q, guard: &'a Guard<'a>) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_hashed(key, guard, self.hash(key))
    }

    pub(crate) fn get_hashed<'a, Q: ?Sized>(
        &'a self,
        key: &Q,
        guard: &'a Guard<'a>,
        hash: u64,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let table = unsafe { guard.protect(&self.table).as_ref()? };

        let bucket_index = bucket_index(hash, table.buckets.len() as _);

        let mut walk = &table.buckets[bucket_index as usize];
        while let Some(node) = unsafe { guard.protect(&walk).as_ref() } {
            if node.hash == hash && node.key.borrow() == key {
                return Some(unsafe { node.value.as_ref() });
            }

            walk = &node.next;
        }

        None
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub(crate) fn contains_key<Q: ?Sized>(&self, key: &Q, guard: &Guard<'_>) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(key, &guard).is_some()
    }

    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        let mut h = self.hash_builder.build_hasher();
        key.hash(&mut h);
        h.finish()
    }
}

// methods that required K/V: Send + Sync, as they potentially call
// `defer_destroy` on keys and values, and the garbage collector
// may execute the destructors on another thread
impl<K, V, S> HashMap<K, V, S>
where
    V: Send + Sync,
    K: Hash + Eq + Send + Sync + Clone,
    S: BuildHasher,
{
    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical.
    pub(crate) fn insert<'a>(&'a self, key: K, value: V, guard: &'a Guard<'a>) -> Option<&'a V> {
        let h = self.hash(&key);
        self.insert_hashed(key, value, guard, h)
    }

    pub(crate) fn insert_hashed<'a>(
        &'a self,
        key: K,
        value: V,
        guard: &'a Guard<'a>,
        hash: u64,
    ) -> Option<&'a V> {
        let mut should_resize = false;

        loop {
            let table_ptr = guard.protect(&self.table);

            let table = match unsafe { table_ptr.as_ref() } {
                Some(table) => table,
                None => self.init_table(None, guard),
            };

            let bucket_index = bucket_index(hash, table.buckets.len() as _);
            let lock_index = lock_index(bucket_index, table.locks.len() as _);

            {
                let _guard = table.locks[lock_index as usize].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !ptr::eq(table, guard.protect(&self.table)) {
                    continue;
                }

                // Try to find this key in the bucket
                let mut node_ptr = &table.buckets[bucket_index as usize];

                loop {
                    let node = guard.protect(&node_ptr);

                    let node_ref = match unsafe { node.as_ref() } {
                        Some(n) => n,
                        None => break,
                    };

                    // If the key already exists, update the value.
                    if hash == node_ref.hash && node_ref.key == key {
                        // Here we create a new node instead of updating the value atomically.
                        //
                        // This is a tradeoff, we don't have to do atomic loads on reads,
                        // but we have to do an extra allocation here for the node.
                        let new_node = Node {
                            key,
                            next: AtomicPtr::new(guard.protect(&node_ref.next)),
                            value: NonNull::from(Box::leak(Box::new(value))),
                            hash,
                        };

                        node_ptr.store(self.collector.link_boxed(new_node), Ordering::Release);

                        let old_val = unsafe { node_ref.value.as_ref() };

                        unsafe {
                            self.collector.retire(node, |mut link| unsafe {
                                let node = link.cast::<Node<K, V>>();
                                let _ = Box::from_raw((*node).value.as_ptr());
                                let _ = Box::from_raw(node);
                            });
                        }

                        return Some(old_val);
                    }

                    node_ptr = &node_ref.next;
                }

                let node = &table.buckets[bucket_index as usize];

                let new = Node {
                    key,
                    next: AtomicPtr::new(guard.protect(&node)),
                    value: NonNull::from(Box::leak(Box::new(value))),
                    hash,
                };

                node.store(self.collector.link_boxed(new), Ordering::Release);

                let count =
                    unsafe { table.counts[lock_index as usize].fetch_add(1, Ordering::AcqRel) + 1 };

                // If the number of elements guarded by this lock has exceeded the budget, resize
                // the hash table.
                //
                // It is also possible that `resize` will increase the budget instead of resizing the
                // hashtable if keys are poorly distributed across buckets.
                if count > self.budget.load(Ordering::SeqCst) {
                    should_resize = true;
                }
            }

            // Resize the table if we're passed our budget.
            //
            // Note that we are not holding the lock anymore.
            if should_resize {
                self.resize(table_ptr, None, guard);
            }

            return None;
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    pub(crate) fn remove<'a, Q: ?Sized>(&'a self, key: &Q, guard: &'a Guard<'a>) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.remove_hashed(key, guard, self.hash(key))
    }

    pub(crate) fn remove_hashed<'a, Q: ?Sized>(
        &'a self,
        key: &Q,
        guard: &'a Guard<'a>,
        hash: u64,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        loop {
            let table = unsafe { guard.protect(&self.table).as_ref()? };

            let bucket_index = bucket_index(hash, table.buckets.len() as _);
            let lock_index = lock_index(bucket_index, table.locks.len() as _);

            {
                let _guard = table.locks[lock_index as usize].lock();

                // If the table just got resized, we may not be holding the right lock.
                if !ptr::eq(table, guard.protect(&self.table)) {
                    continue;
                }

                let mut walk = table.buckets.get(bucket_index as usize);
                while let Some(node_ptr) = walk {
                    let node = guard.protect(&node_ptr);

                    let node_ref = match unsafe { node.as_ref() } {
                        Some(node) => node,
                        None => break,
                    };

                    if hash == node_ref.hash && node_ref.key.borrow() == key {
                        let next = guard.protect(&node_ref.next);
                        node_ptr.store(next, Ordering::Release);

                        unsafe { table.counts[lock_index as usize].fetch_min(1, Ordering::AcqRel) };

                        let val = unsafe { node_ref.value.as_ref() };

                        unsafe {
                            self.collector.retire(node, |mut link| unsafe {
                                let node = link.cast::<Node<K, V>>();
                                let _ = Box::from_raw((*node).value.as_ptr());
                                let _ = Box::from_raw(node);
                            });
                        }

                        return Some(val);
                    }

                    walk = Some(&node_ref.next);
                }
            }

            return None;
        }
    }

    // /// Clears the map, removing all key-value pairs.
    // pub(crate) fn clear(&self, guard: &Guard<'_, Protection>) {
    //     // TODO: acquire each lock and destroy the buckets guarded by it
    //     // sequentially instead of all at once
    //     let (table, _guard) = match self.lock_all(guard) {
    //         Some(x) => x,
    //         // the table was null
    //         None => return,
    //     };

    //     // walk through the table buckets, and set each initial node to null, destroying the rest
    //     // of the nodes and values.
    //     for bucket_ptr in table.buckets.iter() {
    //         let bucket =
    //             unsafe { guard.protect(|| bucket_ptr.load(Ordering::Acquire)) };

    //         if bucket.is_null() {
    //             continue;
    //         }

    //         bucket_ptr.store(ptr::null_mut(), Ordering::Release);

    //         unsafe {
    //             self.collector.retire(bucket, |link| unsafe {
    //                 let node = link.cast::<Node<K, V>>();
    //                 let _: Box<V> = Box::from_raw((*node).value.as_ptr());
    //             });
    //         }

    //         let mut walk = &(*bucket).next;
    //         while let Some(node) = unsafe { walk.load(Ordering::Relaxed, guard).as_ref() } {
    //             unsafe {
    //                 self.collector.retire(node.as_ptr(), |mut link| unsafe {
    //                     let node = link.cast::<Node<K, V>>();
    //                     let _ = Box::from_raw((*node).value.as_ptr());
    //                     let _ = Box::from_raw(node);
    //                 })
    //             }

    //             walk = &node.next;
    //         }
    //     }

    //     // reset the counts
    //     for i in 0..table.counts.len() {
    //         table.counts[i].store(0, Ordering::Release);
    //     }

    //     if let Some(budget) = resize::clear_budget(table.buckets.len(), table.locks.len()) {
    //         // store the new budget, which might be different if the map was
    //         // badly distributed
    //         self.budget.store(budget, Ordering::SeqCst);
    //     }
    // }

    // /// Reserves capacity for at least additional more elements to be inserted in the `HashMap`.
    // pub(crate) fn reserve<'a>(&'a self, additional: usize, guard: &'a Guard<'a, Protection>) {
    //     match unsafe { self.table.load(Ordering::Acquire, guard).as_ref() } {
    //         Some(table) => {
    //             let capacity = table.len() + additional;
    //             self.resize(table, Some(capacity), guard);
    //         }
    //         None => {
    //             self.init_table(Some(additional), guard);
    //         }
    //     }
    // }

    /// Replaces the hash-table with a larger one.
    ///
    /// To prevent multiple threads from resizing the table as a result of races,
    /// the `table` instance that holds the buckets of table deemed too small must
    /// be passed as an argument. We then acquire the resize lock and check if the
    /// table has been replaced in the meantime or not.
    ///
    /// This function may not resize the table if it values are found to be poorly
    /// distributed across buckets. Instead the budget will be increased.
    ///
    /// If a `new_len` is given, that length will be used instead of the default
    /// resizing behavior.
    fn resize<'a>(
        &'a self,
        table: *mut Linked<Table<K, V>>,
        new_len: Option<usize>,
        guard: &'a Guard<'a>,
    ) {
        let _guard = self.resize.lock();

        let table_ref = unsafe { &*table };

        // Make sure nobody resized the table while we were waiting for the resize lock.
        if !ptr::eq(table, guard.protect(&self.table)) {
            // We assume that since the table reference is different, it was
            // already resized (or the budget was adjusted). If we ever decide
            // to do table shrinking, or replace the table for other reasons,
            // we will have to revisit this logic.
            return;
        }

        let current_locks = table_ref.locks.len();

        let (new_len, new_locks_len, new_budget) = match new_len {
            Some(len) => {
                let locks = resize::new_locks(len, current_locks);
                let budget = resize::budget(len, locks.unwrap_or(current_locks));
                (len, locks, budget)
            }
            None => match resize::resize(table_ref.buckets.len(), current_locks, table_ref.len()) {
                resize::Resize::Resize {
                    buckets,
                    locks,
                    budget,
                } => (buckets, locks, budget),
                resize::Resize::DoubleBudget => {
                    let budget = self.budget.load(Ordering::SeqCst);
                    self.budget.store(budget * 2, Ordering::SeqCst);
                    return;
                }
            },
        };

        let mut new_locks = None;

        // Add more locks to account for the new buckets.
        if let Some(len) = new_locks_len {
            let mut locks = Vec::with_capacity(len);

            for i in 0..current_locks {
                locks.push(table_ref.locks[i].clone());
            }

            for _ in current_locks..len {
                locks.push(Arc::new(Mutex::new(())));
            }

            new_locks = Some(locks);
        }

        let new_counts = std::iter::repeat_with(Default::default)
            .take(new_locks_len.unwrap_or(current_locks))
            .collect::<Vec<AtomicUsize>>();

        let mut new_buckets = std::iter::repeat_with(AtomicPtr::default)
            .take(new_len)
            .collect::<Vec<AtomicPtr<_>>>();

        // Now lock the rest of the buckets to make sure nothing is modified while resizing.
        let _guard_rest = table_ref.lock_range(0..table_ref.locks.len());

        // Copy all data into a new buckets, creating new nodes for all elements.
        for i in 0..table_ref.buckets.len() {
            let mut walk = guard.protect(&table_ref.buckets[i]);

            while let Some(node) = unsafe { walk.as_ref() } {
                let new_bucket_index = bucket_index(node.hash, new_len as _);
                let new_lock_index = lock_index(
                    new_bucket_index,
                    new_locks_len.unwrap_or(current_locks) as _,
                );

                let next = guard.protect(&node.next);
                let head = new_buckets[new_bucket_index as usize].load(Ordering::Relaxed);

                // if `next` is the same as the new head, we can skip an allocation and just copy
                // the pointer
                if next == head {
                    new_buckets[new_bucket_index as usize] = AtomicPtr::new(walk);
                } else {
                    new_buckets[new_bucket_index as usize] =
                        AtomicPtr::new(self.collector.link_boxed(Node {
                            key: node.key.clone(),
                            value: node.value.clone(),
                            next: AtomicPtr::new(
                                new_buckets[new_bucket_index as usize].load(Ordering::Relaxed),
                            ),
                            hash: node.hash,
                        }));

                    unsafe {
                        self.collector.retire(walk, move |mut link| unsafe {
                            let _ = Box::from_raw(link.cast::<Node<K, V>>());
                        });
                    }
                }

                new_counts[new_lock_index as usize].fetch_add(1, Ordering::Relaxed);

                walk = next;
            }
        }

        self.budget.store(new_budget, Ordering::SeqCst);

        self.table.store(
            self.collector.link_boxed(Table {
                buckets: new_buckets.into_boxed_slice(),
                locks: new_locks
                    .map(Arc::from)
                    .unwrap_or_else(|| table_ref.locks.clone()),
                counts: new_counts.into_boxed_slice(),
            }),
            Ordering::Release,
        );

        unsafe {
            self.collector.retire(table, move |mut link| unsafe {
                let _ = Box::from_raw(link.cast::<Table<K, V>>());
            });
        }
    }

    /// Acquires all locks for this table.
    fn lock_all<'a>(&'a self, guard: &'a Guard<'a>) -> Option<(&'a Table<K, V>, impl Drop + 'a)> {
        // Acquire the resize lock.
        let _guard = self.resize.lock();

        // Now that we have the resize lock, we can safely load the table and acquire it's
        // locks, knowing that it cannot change.
        let table = unsafe { guard.protect(&self.table).as_ref()? };

        let _guard_rest = table.lock_range(0..table.locks.len());

        struct Guard<A, B>(A, B);

        impl<A, B> Drop for Guard<A, B> {
            fn drop(&mut self) {}
        }

        Some((table, Guard(_guard, _guard_rest)))
    }
}

/// Computes the bucket index for a particular key.
fn bucket_index(hashcode: u64, bucket_count: u64) -> u64 {
    let bucket_index = (hashcode & 0x7fffffff) % bucket_count;
    debug_assert!(bucket_index < bucket_count);

    bucket_index
}

/// Computes the lock index for a particular bucket.
fn lock_index(bucket_index: u64, lock_count: u64) -> u64 {
    let lock_index = bucket_index % lock_count;
    debug_assert!(lock_index < lock_count);

    lock_index
}

impl<K, V, S> Drop for HashMap<K, V, S> {
    fn drop(&mut self) {
        let table = self.table.load(Ordering::Relaxed);

        // table was never allocated
        if table.is_null() {
            return;
        }

        let mut table = unsafe { Box::from_raw(table) };

        // drop all nodes and values
        for bucket in table.buckets.iter_mut() {
            let mut walk = bucket.load(Ordering::Relaxed);
            while let Some(node) = unsafe { walk.as_mut() } {
                let node = unsafe { Box::from_raw(node) };
                let _ = unsafe { Box::from_raw(node.value.as_ptr()) };

                walk = node.next.load(Ordering::Relaxed);
            }
        }
    }
}

unsafe impl<K, V, S> Send for HashMap<K, V, S>
where
    K: Send + Sync,
    V: Send + Sync,
    S: Send,
{
}

unsafe impl<K, V, S> Sync for HashMap<K, V, S>
where
    K: Send + Sync,
    V: Send + Sync,
    S: Sync,
{
}

// impl<K, V, S> Clone for HashMap<K, V, S>
// where
//     K: Sync + Send + Clone + Hash + Eq,
//     V: Sync + Send + Clone,
//     S: BuildHasher + Clone,
// {
//     fn clone(&self) -> HashMap<K, V, S> {
//         let self_pinned = self.pin();
//
//         let clone = Self::with_capacity_and_hasher(self_pinned.len(), self.hash_builder.clone());
//         {
//             let clone_pinned = clone.pin();
//
//             for (k, v) in self_pinned.iter() {
//                 clone_pinned.insert(k.clone(), v.clone());
//             }
//         }
//
//         clone
//     }
// }

// impl<K, V, S> fmt::Debug for HashMap<K, V, S>
// where
//     K: fmt::Debug,
//     V: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let guard = self.collector.thin_shield();
//         f.debug_map().entries(self.iter(&guard)).finish()
//     }
// }

impl<K, V, S> Default for HashMap<K, V, S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

// impl<K, V, S> HashMap<K, V, S>
// where
//     K: Eq + Hash,
//     V: PartialEq,
//     S: BuildHasher,
// {
//     pub(crate) fn eq(
//         &self,
//         guard: &Guard<'_, Protection>,
//         other: &Self,
//         other_guard: &Guard<'_, Protection>,
//     ) -> bool {
//         if self.len(guard) != other.len(other_guard) {
//             return false;
//         }
//
//         self.iter(guard)
//             .all(|(key, value)| other.get(key, other_guard).map_or(false, |v| *value == *v))
//     }
// }
//
// impl<K, V, S> PartialEq for HashMap<K, V, S>
// where
//     K: Eq + Hash,
//     V: PartialEq,
//     S: BuildHasher,
// {
//     fn eq(&self, other: &Self) -> bool {
//         let guard = self.collector.thin_shield();
//         let other_guard = other.collector.thin_shield();
//
//         Self::eq(self, &guard, other, &other_guard)
//     }
// }
//
// impl<K, V, S> Eq for HashMap<K, V, S>
// where
//     K: Eq + Hash,
//     V: Eq,
//     S: BuildHasher,
// {
// }

// /// An iterator over the entries of a `HashMap`.
// ///
// /// This `struct` is created by the [`iter`] method on [`HashMap`]. See its
// /// documentation for more.
// ///
// /// [`iter`]: HashMap::iter
// ///
// /// # Example
// ///
// /// ```
// /// use seize::HashMap;
// ///
// /// let mut map = HashMap::new();
// /// map.insert("a", 1);
// /// let iter = map.iter();
// /// ```
// pub struct Iter<'a, K, V> {
//     table: Shared<'a, Table<K, V>>,
//     guard: &'a Guard<'a, Protection>,
//     current_node: Option<&'a Atomic<Node<K, V>>>,
//     current_bucket: usize,
//     size: usize,
// }
//
// impl<'a, K, V> Iterator for Iter<'a, K, V> {
//     type Item = (&'a K, &'a V);
//
//     fn next(&mut self) -> Option<Self::Item> {
//         loop {
//             let table = match unsafe { self.table.as_ref() } {
//                 Some(table) => table,
//                 None => return None,
//             };
//
//             if let Some(node) = &self
//                 .current_node
//                 .and_then(|n| unsafe { n.load(Ordering::Acquire, self.guard).as_ref() })
//             {
//                 self.size -= 1;
//                 self.current_node = Some(&node.next);
//                 return Some((&node.key, unsafe { node.value.as_ref() }));
//             }
//
//             self.current_bucket += 1;
//
//             if self.current_bucket >= table.buckets.len() {
//                 return None;
//             }
//
//             self.current_node = Some(&table.buckets[self.current_bucket]);
//         }
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         (self.size, Some(self.size))
//     }
// }
//
// impl<K, V> fmt::Debug for Iter<'_, K, V>
// where
//     K: fmt::Debug,
//     V: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_map().entries(self.clone()).finish()
//     }
// }
//
// impl<K, V> Clone for Iter<'_, K, V> {
//     fn clone(&self) -> Self {
//         Self {
//             table: self.table,
//             guard: self.guard,
//             current_node: self.current_node,
//             current_bucket: self.current_bucket,
//             size: self.size,
//         }
//     }
// }
//
// /// An iterator over the keys of a `HashMap`.
// ///
// /// This `struct` is created by the [`keys`] method on [`HashMap`]. See its
// /// documentation for more.
// ///
// /// [`keys`]: HashMap::keys
// ///
// /// # Example
// ///
// /// ```
// /// use seize::HashMap;
// ///
// /// let mut map = HashMap::new();
// /// map.insert("a", 1);
// /// let iter_keys = map.keys();
// /// ```
// pub struct Keys<'a, K, V> {
//     iter: Iter<'a, K, V>,
// }
//
// impl<'a, K, V> Iterator for Keys<'a, K, V> {
//     type Item = &'a K;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next().map(|(k, _)| k)
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }
//
// impl<K, V> fmt::Debug for Keys<'_, K, V>
// where
//     K: fmt::Debug,
//     V: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_list().entries(self.clone()).finish()
//     }
// }
//
// impl<K, V> Clone for Keys<'_, K, V> {
//     fn clone(&self) -> Self {
//         Self {
//             iter: self.iter.clone(),
//         }
//     }
// }
//
// /// An iterator over the values of a `HashMap`.
// ///
// /// This `struct` is created by the [`values`] method on [`HashMap`]. See its
// /// documentation for more.
// ///
// /// [`values`]: HashMap::values
// ///
// /// # Example
// ///
// /// ```
// /// use seize::HashMap;
// ///
// /// let mut map = HashMap::new();
// /// map.insert("a", 1);
// /// let iter_values = map.values();
// /// ```
// pub struct Values<'a, K, V> {
//     iter: Iter<'a, K, V>,
// }
//
// impl<'a, K, V> Iterator for Values<'a, K, V> {
//     type Item = &'a V;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next().map(|(_, v)| v)
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }
//
// impl<K, V> fmt::Debug for Values<'_, K, V>
// where
//     K: fmt::Debug,
//     V: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_list().entries(self.clone()).finish()
//     }
// }
//
// impl<K, V> Clone for Values<'_, K, V> {
//     fn clone(&self) -> Self {
//         Self {
//             iter: self.iter.clone(),
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let map: HashMap<usize, &'static str> = HashMap::new();

        {
            let pinned = map.pin();
            pinned.insert(1, "a");
            pinned.insert(2, "b");
            assert_eq!(pinned.get(&1), Some(&"a"));
            assert_eq!(pinned.get(&2), Some(&"b"));
            assert_eq!(pinned.get(&3), None);

            assert_eq!(pinned.remove(&2), Some(&"b"));
            assert_eq!(pinned.remove(&2), None);

            for i in 0..10000 {
                pinned.insert(i, "a");
            }

            for i in 0..10000 {
                assert_eq!(pinned.get(&i), Some(&"a"));
            }

            // assert!([
            //     pinned.iter().count(),
            //     pinned.len(),
            //     pinned.iter().size_hint().0,
            //     pinned.iter().size_hint().1.unwrap()
            // ]
            // .iter()
            // .all(|&l| l == 10000));
        }

        drop(map);
    }
}
