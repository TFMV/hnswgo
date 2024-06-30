package hnswgo

import (
	"math/rand"
	"testing"
)

func TestNewIndex(t *testing.T) {
	dim := 400
	M := 20
	efConstruction := 10

	batchSize := 100
	maxElements := batchSize * 100

	index := New(dim, M, efConstruction, 432, uint32(maxElements), Cosine, true)
	defer index.Free()

	for i := 0; i < maxElements/batchSize; i++ {
		points, labels := randomPoints(dim, i*batchSize, batchSize)
		index.AddPoints(points, labels, 1, false)
	}
	index.Save("./example.data")
}

func TestLoadIndex(t *testing.T) {
	dim := 400
	batchSize := 100
	maxElements := batchSize * 100

	index := Load("./example.data", Cosine, dim, uint32(maxElements), true)
	defer index.Free()

	// index2 := Load("./example-nonexist.data", Cosine, dim, uint32(maxElements), true)
	// defer index2.Free()
}

func TestVectorSearch(t *testing.T) {
	dim := 400
	batchSize := 100
	maxElements := batchSize * 10000

	index := Load("./example.data", Cosine, dim, uint32(maxElements), true)
	defer index.Free()

	query := genQuery(dim, 10)

	result, err := index.SearchKNN(query, 5, 1)
	if err != nil {
		t.Error(err)
		t.Fail()
		return
	}

	for _, rv := range result {
		if rv == nil {
			t.Fail()
			break
		}
		if rv.Distance <= 0 {
			t.Fail()
		}
		if rv.Label == 0 {
			t.Fail()
		}

		t.Logf("label: %d, distance: %f\n", rv.Label, rv.Distance)
	}

}

func randomPoints(dim int, startLabel int, batchSize int) ([][]float32, []uint32) {
	points := make([][]float32, batchSize)
	labels := make([]uint32, batchSize)

	for i := 0; i < batchSize; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points[i] = v
		labels = append(labels, uint32(startLabel+i))
	}

	return points, labels
}

func genQuery(dim int, size int) [][]float32 {
	points := make([][]float32, size)

	for i := 0; i < size; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points[i] = v
	}

	return points
}
