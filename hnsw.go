package hnswgo

// #cgo CXXFLAGS: -fPIC -pthread -Wall -std=c++11 -O2 -march=native -I.
// #cgo LDFLAGS: -pthread
// #cgo CFLAGS: -I./
// #include <stdlib.h>
// #include "hnsw_wrapper.h"
import "C"
import (
	"errors"
	"unsafe"
)

type SpaceType int

const (
	L2 SpaceType = iota
	IP
	Cosine
)

type HnswIndex struct {
	index *C.HnswIndex
}

type SearchResult struct {
	Label    uint32
	Distance float32
}

func New(dim, M, efConstruction, randSeed int, maxElements uint32, spaceType SpaceType, allowReplaceDeleted bool) *HnswIndex {
	var allowReplace int = 0
	if allowReplaceDeleted {
		allowReplace = 1
	}

	var sType C.spaceType = C.l2
	switch spaceType {
	case L2:
		sType = C.l2
	case IP:
		sType = C.ip
	case Cosine:
		sType = C.cosine
	}

	cindex := C.newIndex(sType, C.int(dim), C.size_t(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.int(allowReplace))

	return &HnswIndex{
		index: cindex,
	}
}

func Load(location string, spaceType SpaceType, dim int, maxElements uint32, allowReplaceDeleted bool) *HnswIndex {
	var allowReplace int = 0
	if allowReplaceDeleted {
		allowReplace = 1
	}

	var sType C.spaceType = C.l2
	switch spaceType {
	case L2:
		sType = C.l2
	case IP:
		sType = C.ip
	case Cosine:
		sType = C.cosine
	}

	cloc := C.CString(location)
	defer C.free(unsafe.Pointer(cloc))

	cindex := C.loadIndex(cloc, sType, C.int(dim), C.size_t(maxElements), C.int(allowReplace))

	return &HnswIndex{
		index: cindex,
	}
}

func (idx *HnswIndex) SetEf(ef int) {
	C.setEf(idx.index, C.size_t(ef))
}

func (idx *HnswIndex) IndexFileSize() uint32 {
	sz := C.indexFileSize(idx.index)

	return uint32(sz)
}

func (idx *HnswIndex) Save(location string) {
	cloc := C.CString(location)
	defer C.free(unsafe.Pointer(cloc))

	C.saveIndex(idx.index, cloc)
}

// Add points to hnsw index.
func (idx *HnswIndex) AddPoints(vectors [][]float32, labels []uint32, concurrency int, replaceDeleted bool) error {
	var replace int = 0
	if replaceDeleted {
		replace = 1
	}

	if len(vectors) <= 0 || len(labels) <= 0 {
		return errors.New("invalid vector data")
	}

	if len(labels) != len(vectors) {
		return errors.New("unmatched vectors size and labels size")
	}

	if len(vectors[0]) != int(idx.index.dim) {
		return errors.New("unmatched dimensions of vector and index")
	}

	rows := len(vectors)
	//as a Go []float32 is layout-compatible with a C float[] so we can pass  Go slice directly to the C function as a pointer to its first element.
	C.addPoints(idx.index,
		(**C.float)(unsafe.Pointer(&vectors[0])),
		C.int(rows),
		(*C.size_t)(unsafe.Pointer(&labels[0])),
		C.int(concurrency),
		C.int(replace))
	return nil
}

func (idx *HnswIndex) SearchKNN(vectors [][]float32, topK int, concurrency int) ([]*SearchResult, error) {
	if len(vectors) <= 0 {
		return nil, errors.New("invalid vector data")
	}

	if len(vectors[0]) != int(idx.index.dim) {
		return nil, errors.New("unmatched dimensions of vector and index")
	}

	if topK > int(idx.index.max_elements) {
		return nil, errors.New("topK is larger than maxElements")
	}

	rows := len(vectors)
	cResult := C.searchKnn(idx.index,
		(**C.float)(unsafe.Pointer(&vectors[0])),
		C.int(rows),
		C.int(topK),
		C.int(concurrency),
	)

	defer C.freeResult(cResult)

	results := make([]*SearchResult, topK) //the resulting slice
	for i := range results {
		r := SearchResult{}
		r.Label = *(*uint32)(unsafe.Add(unsafe.Pointer(cResult.label), i*C.sizeof_ulong))
		r.Distance = *(*float32)(unsafe.Add(unsafe.Pointer(cResult.dist), i*C.sizeof_float))
		results[i] = &r
	}

	return results, nil

}

func (idx *HnswIndex) MarkDeleted(label uint32) {
	C.markDeleted(idx.index, C.size_t(label))
}

func (idx *HnswIndex) UnmarkDeleted(label uint32) {
	C.unmarkDeleted(idx.index, C.size_t(label))
}

func (idx *HnswIndex) ResizeIndex(newSize uint32) {
	C.resizeIndex(idx.index, C.size_t(newSize))
}

// size_t getMaxElements(HnswIndex *index);
// size_t getCurrentCount(HnswIndex *index);
func (idx *HnswIndex) GetMaxElements() uint32 {
	return uint32(C.getMaxElements(idx.index))
}

func (idx *HnswIndex) GetCurrentCount() uint32 {
	return uint32(C.getCurrentCount(idx.index))
}

func (idx *HnswIndex) Free() {
	C.freeHNSW(idx.index)
}

// func Load(location string, dim int, spaceType string) *HNSW {
// 	var hnsw HNSW
// 	hnsw.dim = dim
// 	hnsw.spaceType = spaceType

// 	pLocation := C.CString(location)
// 	if spaceType == "ip" {
// 		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('i'))
// 	} else if spaceType == "cosine" {
// 		hnsw.normalize = true
// 		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('i'))
// 	} else {
// 		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('l'))
// 	}
// 	C.free(unsafe.Pointer(pLocation))
// 	return &hnsw
// }

// func (h *HNSW) Save(location string) {
// 	pLocation := C.CString(location)
// 	C.saveHNSW(h.index, pLocation)
// 	C.free(unsafe.Pointer(pLocation))
// }

// func (h *HNSW) Free() {
// 	C.freeHNSW(h.index)
// }

// func normalizeVector(vector []float32) []float32 {
// 	var norm float32
// 	for i := 0; i < len(vector); i++ {
// 		norm += vector[i] * vector[i]
// 	}
// 	norm = 1.0 / (float32(math.Sqrt(float64(norm))) + 1e-15)
// 	for i := 0; i < len(vector); i++ {
// 		vector[i] = vector[i] * norm
// 	}
// 	return vector
// }

// func (h *HNSW) AddPoint(vector []float32, label uint32) {
// 	if h.normalize {
// 		vector = normalizeVector(vector)
// 	}
// 	C.addPoint(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.ulong(label))
// }

// func (h *HNSW) SearchKNN(vector []float32, N int) ([]uint32, []float32) {
// 	Clabel := make([]C.ulong, N, N)
// 	Cdist := make([]C.float, N, N)
// 	if h.normalize {
// 		vector = normalizeVector(vector)
// 	}
// 	numResult := int(C.searchKnn(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.int(N), &Clabel[0], &Cdist[0]))
// 	labels := make([]uint32, N)
// 	dists := make([]float32, N)
// 	for i := 0; i < numResult; i++ {
// 		labels[i] = uint32(Clabel[i])
// 		dists[i] = float32(Cdist[i])
// 	}
// 	return labels[:numResult], dists[:numResult]
// }

// func (h *HNSW) SetEf(ef int) {
// 	C.setEf(h.index, C.int(ef))
// }
