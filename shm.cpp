#include <type_traits>
#include <memory>
#include <atomic>
#include <stdexcept>
#include <string>
#include <stdint.h>
#include <assert.h>
#include <x86intrin.h>

// Helpers
template<class T>
inline constexpr bool CanMemCopy()
    {return std::is_trivially_copy_assignable<T>::value;}
struct NoHeaderInfo {};
#define LIKELY(cond) __builtin_expect((bool)(cond), 1)

// This is the common base class for the producer and consumer sides.
// Producer and Consumer will derive from this just to hide certain methods.
template< typename T_Object                     // The contained object, main payload
        , typename T_Version    = uint32_t      // Version number of an object
        , typename T_UsrHeader  = NoHeaderInfo  // optional, maybe user needs metadata
        , size_t   A_Alignment  = std::alignment_of<T_Object>::value
        >
class ShmContainerBase
{
    static_assert(CanMemCopy<T_Object>(), "TObject must be mem-copyable");
    struct alignas(A_Alignment) Record;
public:
    // In a 64-bit process, the capacity can be quite huge: 1-256 TB.
    // Neither physical memory nor disk space will not be consumed
    // until data is actually written to it.
    enum class eRole { PRODUCER, CONSUMER }; // for checking API usage
    ShmContainerBase(size_t capacity_num_records, std::string file_path, eRole);

    // API: Guranteed consistent, atomic read.
    struct VersionUnchecked : std::exception {};
    class ScopedConsume;
    ScopedConsume consume_begin(size_t obj_index)
        {return ScopedConsume(&m_shared_mem->records[obj_index]);}

    // API: Atomically update a record.
    class ScopedProduce;
    ScopedProduce produce_begin(size_t obj_index)
        {return ScopedProduce(&m_shared_mem->records[obj_index]);}

    ScopedProduce emplace_back()
        { return produce_begin(m_shared_mem->hdr.size++);}

    // Convenience method
    void push_back(T_Object const& obj)
        {
        auto prod = emplace_back();
        *prod = obj;
        prod.produce_commit();
        }

    // Optional. Maybe user needs to add meta-data to the container,
    T_UsrHeader& user_header() {return m_shared_mem->hdr.user_header;}

public: // Boilerplate standard container interface

    size_t size() const;
    size_t capacity() const;

    class iterator;
    class const_iterator;
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    ShmContainerBase(ShmContainerBase const&) = default; // shared reference copy
    ShmContainerBase(ShmContainerBase&&) = default;
    ShmContainerBase() = default;

private:
    using vsize_t    = size_t; // Single-producer only
    using version_t  = std::atomic<T_Version>;
    using refcount_t = std::atomic<size_t>;
    using has_prod_t = std::atomic<bool>;
    static constexpr T_Version INVALID_VERSION = 0;

    struct alignas(64) Header
    {
        vsize_t     size {};
        vsize_t     capacity {};
        version_t   accumulated_version {}; // increments when any record does
        refcount_t  refcount {}; // producer + consumers
        bool        delete_file_after_last_ref {};
        has_prod_t  has_producer {}; // single-producer check
        T_UsrHeader user_header {};
    };

    struct alignas(A_Alignment) Record
    {
        T_Object    payload {};
        version_t   version_a {INVALID_VERSION};
        version_t   version_b {INVALID_VERSION};

        T_Version   cons_begin() const        {return version_a.load(std::memory_order_acquire);}
        T_Version   cons_commit() const       {return version_b.load(std::memory_order_acquire);}
        T_Version   prod_begin()              {return ++version_b;}
        void        prod_commit(T_Version vv) {version_a.store(vv, std::memory_order_release);}
    };

    struct MemLayout
    {
        Header      hdr;
        Record      records[];
    };

private:
    std::shared_ptr<MemLayout>  m_shared_mem; // mmap() & munmap()
};

//==============================================================================
template< typename T_Object, typename T_Version, typename UsrHdr, size_t Align>
class ShmContainerBase<T_Object, T_Version, UsrHdr, Align>::
ScopedConsume
{
    Record*           m_rec {};
    T_Version mutable m_pre_consume_ver {INVALID_VERSION};
public:
    explicit ScopedConsume(Record* p = nullptr) : m_rec(p) {}
    ~ScopedConsume() {if(m_rec) throw VersionUnchecked();} // User forgot check
    bool try_consume_commit()
        {
        assert(m_rec);
        auto const curr_ver = m_rec->cons_commit();
        if(LIKELY(curr_ver == m_pre_consume_ver))
            {
            cancel_consume(); // prevent exception
            return true;
            }
        m_pre_consume_ver = curr_ver;
        return false; // user shall now retry consume the object
        }
    T_Object const* get() const __attribute__((const))
        {
        assert(m_rec);
        // The first get() call marks beginning of consumption. Remember version
        if(INVALID_VERSION == m_pre_consume_ver)
            m_pre_consume_ver = m_rec->cons_begin();
        return &m_rec->payload;
        }
    T_Object get_copy()
        {
        T_Object res;
        do { res = *get(); } while(!this->try_consume_commit());
        return res;
        }
    explicit operator bool() const {return !!m_rec;}
    T_Object const* operator->() const {return get();}
    T_Object const& operator*() const {return *get();}
    bool operator==(ScopedConsume const& rhs) const {return m_rec == rhs.m_rec;}
    bool operator!=(ScopedConsume const& rhs) const {return m_rec != rhs.m_rec;}
private:
    void adv() {assert(m_rec); ++m_rec;}
    void cancel_consume() {m_rec = nullptr;}
};

//==============================================================================
template< typename T_Object, typename T_Version, typename UsrHdr, size_t Align>
class ShmContainerBase<T_Object, T_Version, UsrHdr, Align>::
ScopedProduce
{
    Record*     m_rec {};
    T_Version   m_initial_ver {INVALID_VERSION};
public:
    explicit ScopedProduce(Record* p = nullptr) : m_rec(p) {}
    ~ScopedProduce()       {if(m_rec) produce_commit(); } // auto-commit, can't fail
    T_Object* operator->() {return get();}
    T_Object& operator*()  {return *get();}
    T_Object* get() __attribute__((const))
        {
        assert(m_rec);
        if(INVALID_VERSION == m_initial_ver)
            m_initial_ver = m_rec->prod_begin();
        return &m_rec->payload;
        }
    void produce_commit(bool const a_used_memcpy_or_movnti = true)
        {
        if(a_used_memcpy_or_movnti)
            _mm_sfence();
        m_rec->prod_commit(m_initial_ver);
        m_rec = nullptr;
        }
};

//==============================================================================
template< typename T_Object, typename Ver, typename UsrHdr, size_t Align>
class ShmContainerBase<T_Object, Ver, UsrHdr, Align>::iterator
{
    ScopedConsume m_rec_ptr {};
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T_Object;
    using pointer           = T_Object*;
    using reference         = T_Object&;

    iterator& operator++()   {assert(m_rec_ptr); m_rec_ptr.adv(); return *this;}
    iterator  operator++(int){auto tmp = m_rec_ptr; m_rec_ptr.adv(); return tmp;}
    T_Object operator*() // Note: returns copy rather than reference
    {
        assert(m_rec_ptr);
        T_Object local_copy;
        auto scoped_version_check = m_rec_ptr->consume_begin();
        do {
            local_copy = m_rec_ptr->payload;
        } while(m_rec_ptr->try_consume_commit());
        return *m_rec_ptr;
    }
    // pointer   operator->()   {assert(m_rec_ptr); return m_rec_ptr;}
};

//==============================================================================
template< typename T_Object
        , typename T_Version    = uint32_t
        , typename T_UsrHeader  = NoHeaderInfo // optional
        , size_t   A_Alignment  = std::alignment_of<T_Object>::value
        >
struct ShmContainerProducer
    : private ShmContainerBase<T_Object, T_Version, T_UsrHeader, A_Alignment>
{
    using Base = ShmContainerBase<T_Object, T_Version, T_UsrHeader, A_Alignment>;
    using Base::produce_begin;
    using Base::emplace_back;
    ShmContainerProducer(size_t capacity_num_records, std::string file_path)
        : Base(capacity_num_records, file_path, Base::eRole::PRODUCER)
        {}
};

//==============================================================================
template< typename T_Object
        , typename T_Version    = uint32_t
        , typename T_UsrHeader  = NoHeaderInfo // optional
        , size_t   A_Alignment  = std::alignment_of<T_Object>::value
        >
struct ShmContainerConsumer
    : private ShmContainerBase<T_Object, T_Version, T_UsrHeader, A_Alignment>
{
    using Base = ShmContainerBase<T_Object, T_Version, T_UsrHeader, A_Alignment>;
    using Base::consume_begin;
    ShmContainerConsumer(size_t capacity_num_records, std::string file_path)
        : Base(capacity_num_records, file_path, Base::eRole::CONSUMER)
        {}
};

//==============================================================================

// Example contained object
struct NseTicker
{
    uint32_t ask_px;
    uint32_t ask_qx;
    uint32_t bid_px;
    uint32_t bid_qx;
};

void example_producer()
{
    ShmContainerProducer<NseTicker> shm_container(1000, "/tmp/nse_tickers.shm");

    // Get a "versioned pointer"
    auto vptr = shm_container.emplace_back();

    // Update the object in any way you want (not atomically),
    // but be quick because consumers will keep retrying their read
    // until vptr gets out of scope or vptr.produce_commit() is called.
    vptr->bid_px = 39000;
    vptr->ask_px = 41000;
    vptr->bid_qx = 55;
    vptr->ask_qx = 77;

    // optional. Will automatically be called when vptr gets out of scope anyway
    vptr.produce_commit();

    // Now consumers will see the updated version only
}

void example_consumer()
{
    // Consumers "connect" to the shared memory by passing in the file path.
    ShmContainerConsumer<NseTicker> shm_container(1000, "/tmp/nse_tickers.shm");

    // The "versioned pointer" makes sure you can only do consistent reads.
    // Here we are beginning a read of object [0] in the container.
    auto vptr = shm_container.consume_begin(0);
    NseTicker const ticker = vptr.get_copy();

}
