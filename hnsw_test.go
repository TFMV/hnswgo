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
		points, labels := randomPoint(dim, i*batchSize, batchSize)
		index.AddPoints(points, labels, 1, false)
	}

	index.Save("example.data")
}

func randomPoint(dim int, startLabel int, batchSize int) ([][]float32, []uint32) {
	points := make([][]float32, batchSize)
	labels := make([]uint32, batchSize)

	for i := 0; i < batchSize; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points = append(points, v)
		labels = append(labels, uint32(startLabel+i))
	}

	return points, labels
}
