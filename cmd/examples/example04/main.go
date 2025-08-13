// This example shows you how to use MongoDB as a vector database to perform
// a nearest neighbor vector search. The example will create a vector search
// index, store 2 documents, and perform a vector search.
//
// # Running the example:
//
//  $ make example4
//
// # This requires running the following command:
//
//	$ make compose-up // This starts MongoDB and OpenWebUI in docker compose.
//
// # You can use this command to open a prompt to mongodb:
//
//  $ make mongo

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type document struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
}

type searchResult struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
	Score     float64   `bson:"score"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------
	// Connect to mongo

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("connectToMongo: %w", err)
	}
	defer client.Disconnect(ctx)

	fmt.Println("\nConnected to MongoDB")

	// -------------------------------------------------------------------------
	// Create database and collection

	const dbName = "example4"
	const collectionName = "book"

	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return fmt.Errorf("createCollection: %w", err)
	}

	fmt.Println("Created Collection")

	// -------------------------------------------------------------------------
	// Create vector index

	const indexName = "vector_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: 4,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return fmt.Errorf("createVectorIndex: %w", err)
	}

	fmt.Println("Created Vector Index")

	// -------------------------------------------------------------------------
	// Apply a unique index just to be safe.

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "id", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	col.Indexes().CreateOne(ctx, indexModel)

	fmt.Println("Created ID Index")

	// -------------------------------------------------------------------------
	// Delete any documents that might be there.

	col.DeleteOne(ctx, bson.D{{Key: "id", Value: 1}})
	col.DeleteOne(ctx, bson.D{{Key: "id", Value: 2}})

	// -------------------------------------------------------------------------
	// Store some documents with their embeddings.

	fmt.Println("\nInserting Documents")

	d1 := document{
		ID:        1,
		Text:      "this is text 1",
		Embedding: []float64{1.0, 2.0, 3.0, 4.0},
	}

	d2 := document{
		ID:        2,
		Text:      "this is text 2",
		Embedding: []float64{1.5, 2.5, 3.5, 4.5},
	}

	res, err := col.InsertMany(ctx, []any{d1, d2})
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	fmt.Printf("%v\n", res.InsertedIDs)

	// We need to give Mongo a little time to index the documents.
	time.Sleep(time.Second)

	// -------------------------------------------------------------------------
	// Perform a vector search.

	fmt.Print("\n---- VECTOR SEARCH ----\n\n")

	results, err := vectorSearch(ctx, col, []float64{1.2, 2.2, 3.2, 4.2}, 10)
	if err != nil {
		return fmt.Errorf("storeDocuments: %w", err)
	}

	fmt.Printf("%#v\n", results)

	return nil
}

func vectorSearch(ctx context.Context, col *mongo.Collection, vector []float64, limit int) ([]searchResult, error) {

	// -------------------------------------------------------------------------
	// Example MongoDB query

	/*
		db.book.aggregate([
			{
				"$vectorSearch": {
					"index": "vector_index",
					"exact": false,
					"numCandidates": 10,
					"path": "embedding",
					"queryVector": [1.2, 2.2, 3.2, 4.2],
					"limit": 10
				}
			},
			{
				"$project": {
					"text": 1,
					"embedding": 1,
					"score": {
						"$meta": "vectorSearchScore"
					}
				}
			}
		])
	*/

	// -------------------------------------------------------------------------
	// We want to find the nearest neighbors from the specified vector.

	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":         "vector_index",
				"exact":         false,
				"path":          "embedding",
				"queryVector":   vector,
				"numCandidates": limit,
				"limit":         limit,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"id":        1,
				"text":      1,
				"embedding": 1,
				"score": bson.M{
					"$meta": "vectorSearchScore",
				},
			}},
		},
	}

	cur, err := col.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("aggregate: %w", err)
	}
	defer cur.Close(ctx)

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}
