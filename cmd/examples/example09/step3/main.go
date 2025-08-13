// This example takes step2 and shows you how to store the image path and its
// vector embedding into a vector database.
//
// # Running the example:
//
//	$ make example9-step3
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
//	$ make compose-up // This starts the MongoDB service.
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/dsoprea/go-exif/v3"
	exifcommon "github.com/dsoprea/go-exif/v3/common"
	jpg "github.com/dsoprea/go-jpeg-image-structure/v2"
	pis "github.com/dsoprea/go-png-image-structure/v2"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	url            = "http://localhost:11434"
	model          = "qwen2.5vl:latest"
	imagePath      = "cmd/samples/gallery/roseimg.png"
	embedModel     = "bge-m3:latest"
	dbName         = "example9"
	collectionName = "images-3"
)

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float32 `bson:"embedding"`
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	// -------------------------------------------------------------------------

	llm, err := ollama.New(
		ollama.WithModel(model),
		ollama.WithServerURL(url),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	llmEmbed, err := ollama.New(
		ollama.WithModel(embedModel),
		ollama.WithServerURL(url),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	// -------------------------------------------------------------------------

	mongoClient, err := initDatabase(dbName, collectionName)
	if err != nil {
		return fmt.Errorf("initDatabase: %w", err)
	}

	col := mongoClient.Database(dbName).Collection(collectionName)

	// -------------------------------------------------------------------------

	findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: imagePath}})
	if findRes.Err() == nil {
		fmt.Println("Delete existing image from database")
		_, err := col.DeleteOne(ctx, bson.D{{Key: "file_name", Value: imagePath}})
		if err != nil {
			return fmt.Errorf("delete image: %w", err)
		}
	}

	// -------------------------------------------------------------------------

	data, mimeType, err := processImage(imagePath)
	if err != nil {
		return fmt.Errorf("process image: %w", err)
	}

	// ---------------------------------------------------------------------

	fmt.Print("\nGenerating image description:\n\n")

	prompt := `Describe the image. Be concise and accurate. Do not be overly
		verbose or stylistic. Make sure all the elements in the image are
		enumerated and described. Do not include any additional details. Keep
		the description under 200 words. At the end of the description, create
		a list of tags with the names of all the elements in the image. Do not
		output anything past this list.
		Encode the list as valid JSON, as in this example:
		[
			"tag1",
			"tag2",
			"tag3",
			...
		]
		Make sure the JSON is valid, doesn't have any extra spaces, and is
		properly formatted.`

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryContent{
					MIMEType: mimeType,
					Data:     data,
				},
				llms.TextContent{
					Text: prompt,
				},
			},
		},
	}

	cr, err := llm.GenerateContent(
		ctx,
		messages,
		llms.WithMaxTokens(500),
		llms.WithTemperature(1.0),
	)
	if err != nil {
		return fmt.Errorf("generate content: %w", err)
	}

	fmt.Print(cr.Choices[0].Content)
	fmt.Print("\n\n")

	// ---------------------------------------------------------------------

	fmt.Print("Updating image description:\n\n")

	if err := updateImage(imagePath, cr.Choices[0].Content); err != nil {
		return fmt.Errorf("update image: %w", err)
	}

	// ---------------------------------------------------------------------

	fmt.Print("Generate embeddings for the image description:\n\n")

	vectors, err := llmEmbed.CreateEmbedding(ctx, []string{cr.Choices[0].Content})
	if err != nil {
		return fmt.Errorf("create embedding: %w", err)
	}

	fmt.Printf("%v...%v\n\n", vectors[0][0:3], vectors[0][len(vectors[0])-3:])

	// ---------------------------------------------------------------------

	fmt.Print("Inserting image description into the database:\n\n")

	d1 := document{
		FileName:    imagePath,
		Description: cr.Choices[0].Content,
		Embedding:   vectors[0],
	}

	res, err := col.InsertOne(ctx, d1)
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	fmt.Printf("Inserted db id: %s\n\n", res.InsertedID)

	fmt.Println("DONE")
	return nil
}

func initDatabase(dbName string, collectionName string) (*mongo.Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------
	// Connect to mongo

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	fmt.Println("\nConnected to MongoDB")

	// -------------------------------------------------------------------------
	// Create database and collection

	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	fmt.Println("Created Collection")

	// -------------------------------------------------------------------------
	// Create vector index

	const indexName = "vector_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: 1024,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	fmt.Println("Created Vector Index")

	// -------------------------------------------------------------------------
	// Apply a unique index just to be safe.

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "file_name", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	if _, err := col.Indexes().CreateOne(ctx, indexModel); err != nil {
		return nil, fmt.Errorf("createUniqueIndex: %w", err)
	}

	fmt.Println("Created Unique file_name Index")

	return client, nil
}

func processImage(fileName string) ([]byte, string, error) {
	data, err := readImage(fileName)
	if err != nil {
		return nil, "", fmt.Errorf("read image: %w", err)
	}

	var mimeType string
	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		mimeType = "image/jpg"
	case ".png":
		mimeType = "image/png"
	default:
		return nil, "", fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	return data, mimeType, nil
}

func readImage(fileName string) ([]byte, error) {
	f, err := os.OpenFile(fileName, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	return data, nil
}

func updateImage(fileName string, description string) error {
	im, err := exifcommon.NewIfdMappingWithStandard()
	if err != nil {
		return fmt.Errorf("new idf mapping: %w", err)
	}

	ti := exif.NewTagIndex()
	ib := exif.NewIfdBuilder(im, ti, exifcommon.IfdStandardIfdIdentity, exifcommon.EncodeDefaultByteOrder)

	err = ib.AddStandardWithName("ImageDescription", description)
	if err != nil {
		return fmt.Errorf("add standard: %w", err)
	}

	// -------------------------------------------------------------------------

	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		intfc, err := jpg.NewJpegMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*jpg.SegmentList)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.Write(f)
		if err != nil {
			return fmt.Errorf("write: %w", err)
		}
		defer f.Close()

	case ".png":
		intfc, err := pis.NewPngMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*pis.ChunkSlice)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.WriteTo(f)
		if err != nil {
			return fmt.Errorf("write: %w", err)
		}
		defer f.Close()

	default:
		return fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	return nil
}
