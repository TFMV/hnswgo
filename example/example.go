package main

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/oligo/hnswgo"
)

func main() {
	dim := 400
	M := 20
	efConstruction := 10

	batchSize := 1000
	maxElements := batchSize * 100

	var index *hnswgo.HnswIndex
	if PathExists("./example.data") {
		index = hnswgo.Load("./example.data", hnswgo.Cosine, dim, uint32(maxElements), true)
	} else {
		start := time.Now()
		index = hnswgo.New(dim, M, efConstruction, 432, uint32(maxElements), hnswgo.Cosine, true)
		for i := 0; i < maxElements/batchSize; i++ {
			points, labels := randomPoints(dim, i*batchSize, batchSize)
			index.AddPoints(points, labels, 4, false)
		}

		defer index.Save("example.data")
		fmt.Printf("Time elapsed: %f", time.Since(start).Seconds())
	}

	defer index.Free()

	// query := [][]float32{randomPoint(dim)}
	// result, err := index.SearchKNN(query, 5, 1)
	// if err != nil {
	// 	panic(err)
	// }

	// for _, rv := range result {
	// 	fmt.Printf("label: %d, distance: %f\n", rv.Label, rv.Distance)
	// }

}

func randomPoint(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
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

func PathExists(path string) bool {
	stat, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false
	}

	if err == nil || stat != nil {
		return true
	}

	return false
}
