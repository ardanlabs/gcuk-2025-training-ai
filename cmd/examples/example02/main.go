package main

import (
	"context"
	"fmt"
	"log"

	"github.com/ardanlabs/ai-training/foundation/vector"
	"github.com/tmc/langchaingo/llms/ollama"
)

const (
	url   = "http://localhost:11434"
	model = "bge-m3:latest"
)

// =============================================================================

type data struct {
	Name      string
	Text      string
	Embedding []float32 // The vector where the data is embedded in space.
}

// Vector can convert the specified data into a vector.
func (d data) Vector() []float32 {
	return d.Embedding
}

func main() {

	llm, err := ollama.New(
		ollama.WithModel(model),
		ollama.WithServerURL(url),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Old way of representing this data with our own vector data points.
	// dataPoints := []vector.Data{
	// 	data{Name: "Horse   ", Authority: 0.0, Animal: 1.0, Human: 0.0, Rich: 0.0, Gender: +1.0},
	// 	data{Name: "Man     ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: -1.0},
	// 	data{Name: "Woman   ", Authority: 0.0, Animal: 0.0, Human: 1.0, Rich: 0.0, Gender: +1.0},
	// 	data{Name: "King    ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: -1.0},
	// 	data{Name: "Queen   ", Authority: 1.0, Animal: 0.0, Human: 1.0, Rich: 1.0, Gender: +1.0},
	// }

	// Apply the feature vectors to the hand crafted data points.
	// This time you need to use words since we are using a word based model.
	dataPoints := []vector.Data{
		data{Name: "Horse   ", Text: "Animal, Female"},
		data{Name: "Man     ", Text: "Human,  Male,   Pants, Poor, Worker"},
		data{Name: "Woman   ", Text: "Human,  Female, Dress, Poor, Worker"},
		data{Name: "King    ", Text: "Human,  Male,   Pants, Rich, Ruler"},
		data{Name: "Queen   ", Text: "Human,  Female, Dress, Rich, Ruler"},
	}

	fmt.Print("\n")

	// Iterate over each data point and use the LLM to generate the vector
	// embedding related to the model.
	for i, dp := range dataPoints {
		dataPoint := dp.(data)

		vectors, err := llm.CreateEmbedding(context.Background(), []string{dataPoint.Text})
		if err != nil {
			log.Fatal(err)
		}

		dataPoint.Embedding = vectors[0]
		dataPoints[i] = dataPoint

		fmt.Printf("Vector: Name(%s) len(%d) %v...%v\n", dataPoint.Name, len(vectors[0]), vectors[0][0:2], vectors[0][len(vectors[0])-2:])
	}

	// -------------------------------------------------------------------------

	// Compare each data point to every other by performing a cosine
	// similarity comparison using the vector embedding from the LLM.
	for _, target := range dataPoints {
		results := vector.Similarity(target, dataPoints...)

		for _, result := range results {
			fmt.Printf("%s -> %s: %.2f%% similar\n",
				result.Target.(data).Name,
				result.DataPoint.(data).Name,
				result.Percentage)
		}
		fmt.Print("\n")
	}

	// -------------------------------------------------------------------------

	// Perform the same vector math as in example2 using the LLM vector embedding.

	// You can perform vector math by adding and subtracting vectors.
	kingSubMan := vector.Sub(dataPoints[3].Vector(), dataPoints[1].Vector())
	kingSubManPlusWoman := vector.Add(kingSubMan, dataPoints[2].Vector())
	queen := dataPoints[4].Vector()

	// Now compare a (King - Man + Woman) to a Queen.
	result := vector.CosineSimilarity(kingSubManPlusWoman, queen)
	fmt.Printf("King - Man + Woman ~= Queen similarity: %.2f%%\n", result*100)
}
