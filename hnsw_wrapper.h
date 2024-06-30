// hnsw_wrapper.h
#ifdef __cplusplus
extern "C"
{
#endif

    typedef void *HNSW;
    typedef void *HnswSpace;
    typedef enum {
        l2, ip, cosine
    } spaceType;

    typedef struct
    {
        HNSW hnsw;
        HnswSpace space;
        spaceType space_type;
        int dim;
        int max_elements;
        int allow_replace_deleted;
        int normalize;
    } HnswIndex;



    //typedef bool (*filter_func)(int label);

    typedef struct
    {
        unsigned long int *label;
        float *dist;
    } SearchResult;

    HnswIndex *newIndex(spaceType space_type, const int dim, size_t max_elements, int M, int ef_construction, int rand_seed, int allow_replace_deleted);
    void setEf(HnswIndex *index, size_t ef);
    size_t indexFileSize(HnswIndex *index);
    void saveIndex(HnswIndex *index, char *location);
    HnswIndex *loadIndex(char *location, spaceType space_type, int dim, size_t max_elements, int allow_replace_deleted);
    void addPoints(HnswIndex *index, float **vectors, int rows, size_t *labels, int num_threads, int replace_deleted);
    void markDeleted(HnswIndex *index, size_t label);
    void unmarkDeleted(HnswIndex *index, size_t label);
    void resizeIndex(HnswIndex *index, size_t new_size);
    size_t getMaxElements(HnswIndex *index);
    size_t getCurrentCount(HnswIndex *index);
    // SearchResult *searchKnn(HnswIndex *index, float **vectors, int rows, int k, filter_func filter, int num_threads);
    SearchResult *searchKnn(HnswIndex *index, float **vectors, int rows, int k, int num_threads);

    void freeHNSW(HnswIndex *index);
    void freeResult(SearchResult *result);

// HNSW loadHNSW(char *location, int dim, char stype);
//   HNSW saveHNSW(HNSW index, char *location);
//   void freeHNSW(HNSW index);
//   void addPoint(HNSW index, float *vec, unsigned long int label);
//   int searchKnn(HNSW index, float *vec, int N, unsigned long int *label, float *dist);
//   void setEf(HNSW index, int ef);
#ifdef __cplusplus
}
#endif
