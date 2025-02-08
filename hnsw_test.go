package hnswgo

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

const testVectorDB = "./test.db"
const (
	dim            = 400
	M              = 20
	efConstruction = 10

	batchSize = 100
)

func newTestIndex(batch int, allowRepaceDeleted bool) *HnswIndex {
	maxElements := batch * batchSize

	index := New(dim, M, efConstruction, 55, uint64(maxElements), Cosine, allowRepaceDeleted)

	for i := 0; i < batch; i++ {
		points, labels := randomPoints(dim, i*batchSize, batchSize)
		index.AddPoints(points, labels, 1, false)
	}

	return index
}

func TestNewIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	idx := newTestIndex(1, true)
	defer idx.Free()

	if idx.GetMaxElements() != maxElements {
		t.Fail()
	}

	if idx.GetAllowReplaceDeleted() != true {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
		t.Fail()
	}

}

func TestLoadAndSaveIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	// setup
	idx := newTestIndex(1, true)
	idx.Save(testVectorDB)
	idx.Free()

	index, err := Load(testVectorDB, Cosine, dim, uint64(maxElements), true)
	if err != nil {
		t.Fail()
	}
	index.SetEf(efConstruction)
	defer index.Free()

	index.Save(testVectorDB)
	t.Cleanup(func() {
		deleteDB()
	})
}

func TestResizeIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	idx := newTestIndex(1, false)
	defer idx.Free()

	if idx.GetMaxElements() != maxElements {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
		t.Fail()
	}

	if idx.GetAllowReplaceDeleted() != false {
		t.Fail()
	}

	points, labels := randomPoints(dim, 1*batchSize, batchSize)
	err := idx.AddPoints(points, labels, 1, false)
	if err == nil {
		t.Log(err)
		t.FailNow()
	}

	idx.ResizeIndex(maxElements * 2)
	if idx.GetMaxElements() != maxElements*2 {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
		t.Fail()
	}

	err = idx.AddPoints(points, labels, 1, false)
	if err != nil {
		t.Log(err)
		t.Fail()
	}
}

func TestReplacePoint(t *testing.T) {
	allowRepaceDeleted := true
	maxElements := 100
	index := New(dim, M, efConstruction, 505, uint64(maxElements), Cosine, allowRepaceDeleted)
	defer index.Free()

	if !index.GetAllowReplaceDeleted() {
		t.Fail()
	}

	points, labels := randomPoints(dim, 0, maxElements)
	index.AddPoints(points, labels, 1, false)

	index.MarkDeleted(labels[len(labels)-1])

	err := index.AddPoints([][]float32{randomPoint(dim)}, []uint64{math.MaxUint64 - 1}, 1, false)
	if err == nil {
		t.Fail()
	}

	err = index.AddPoints([][]float32{randomPoint(dim)}, []uint64{math.MaxUint64 - 1}, 1, true)
	if err != nil {
		t.Fail()
	}

}

func TestVectorSearch(t *testing.T) {
	indexPath := "testdata/index.hnsw"

	// Ensure testdata directory exists
	if err := os.MkdirAll("testdata", os.ModePerm); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}

	// Create a new index if it doesn't exist
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		fmt.Println("Index file does not exist. Creating a new one...")

		index := New(3, 16, 200, 42, 10000, Cosine, true)

		// Add some sample data - these vectors are only dimension 3!
		vectors := [][]float32{
			{0.1, 0.2, 0.3}, // dimension 3
			{0.4, 0.5, 0.6}, // dimension 3
			{0.7, 0.8, 0.9}, // dimension 3
		}
		labels := []uint64{1, 2, 3}

		err := index.AddPoints(vectors, labels, 2, true)
		if err != nil {
			t.Fatalf("Failed to add points: %v", err)
		}

		// Save the index
		index.Save(indexPath)
		fmt.Println("Index created and saved at:", indexPath)
	}

	// Load the saved index
	index, err := Load(indexPath, Cosine, 3, 10000, true)
	if err != nil {
		t.Fatalf("Failed to load index: %v", err)
	}

	// Perform a search
	query := [][]float32{
		{0.3, 0.3, 0.3},
	}
	results, err := index.SearchKNN(query, 2, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Search results:", results)
}

func TestGetVectorData(t *testing.T) {
	indexPath := "testdata/index.hnsw"

	// Ensure testdata directory exists
	if err := os.MkdirAll("testdata", os.ModePerm); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}

	// Create a new index if it doesn't exist
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		fmt.Println("Index file does not exist. Creating a new one...")

		index := New(3, 16, 200, 42, 10000, Cosine, true)

		vectors := [][]float32{
			{0.1, 0.2, 0.3}, // dimension 3
			{0.4, 0.5, 0.6}, // dimension 3
			{0.7, 0.8, 0.9}, // dimension 3
		}
		labels := []uint64{1, 2, 3}

		err := index.AddPoints(vectors, labels, 2, true)
		if err != nil {
			t.Fatalf("Failed to add points: %v", err)
		}

		index.Save(indexPath)
		fmt.Println("Index created and saved at:", indexPath)
		t.Cleanup(func() {
			deleteDB()
		})
	}

	index, err := Load(indexPath, Cosine, 3, 10000, true)
	if err != nil {
		t.Fatalf("Failed to load index: %v", err)
	}
	defer index.Free()

	vector := index.GetDataByLabel(1)
	fmt.Println("Vector data:", vector)
}

func randomPoints(dim int, startLabel int, batchSize int) ([][]float32, []uint64) {
	points := make([][]float32, batchSize)
	labels := make([]uint64, 0)

	for i := 0; i < batchSize; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points[i] = v
		labels = append(labels, uint64(startLabel+i))
	}

	return points, labels
}

func pathExists(path string) bool {
	stat, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false
	}

	if err == nil || stat != nil {
		return true
	}

	return false
}

func deleteDB() error {
	if pathExists(testVectorDB) {
		return os.Remove(testVectorDB)
	}

	return nil
}

func randomPoint(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

func _genQuery(dim int, size int) [][]float32 {
	// Implementation of _genQuery function
	return nil // Placeholder return, actual implementation needed
}
