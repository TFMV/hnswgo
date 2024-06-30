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

	maxElements := 10000

	var index *hnswgo.HnswIndex
	if PathExists("./example.data") {
		index = hnswgo.Load("./example.data", hnswgo.Cosine, dim, uint32(maxElements), true)
	} else {
		start := time.Now()
		index = hnswgo.New(dim, M, efConstruction, 432, uint32(maxElements), hnswgo.Cosine, true)
		for i := 0; i < maxElements/2; i++ {
			index.AddPoints(randomPoint(dim), uint32(i))
		}

		index.Save("example.data")
		fmt.Printf("Time elapsed: %f", time.Since(start).Seconds())
	}

	defer index.Free()

	// labels, vectors := index.SearchKNN(randomPoint(dim), 10)
	// for i, lb := range labels {
	// 	fmt.Printf("label: %d, vector: %v\n", lb, vectors[i])
	// }

}

func randomPoint(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
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
