package main

import (
	"bufio"
	"context"
	"encoding/json"
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
	embedModel     = "bge-m3:latest"
	dbName         = "example9"
	collectionName = "images-5"
	gallaryPath    = "cmd/samples/gallery/"
)

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float32 `bson:"embedding"`
}

type searchResult struct {
	FileName    string    `bson:"file_name" json:"file_name"`
	Description string    `bson:"description" json:"image_description"`
	Embedding   []float32 `bson:"embedding" json:"-"`
	Score       float64   `bson:"score" json:"-"`
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

	// -------------------------------------------------------------------------

	files, err := getFilesFromDirectory(gallaryPath)
	if err != nil {
		return fmt.Errorf("get files: %w", err)
	}

	for _, fileName := range files {
		fmt.Printf("\nProcessing image: %s\n", fileName)

		findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: fileName}})
		if findRes.Err() == nil {
			fmt.Println("  - Image already exists")
			continue
		}

		data, mimeType, err := processImage(fileName)
		if err != nil {
			return fmt.Errorf("process image: %w", err)
		}

		fmt.Println("  - Generating image description")

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

		// -------------------------------------------------------------------------

		fmt.Println("  - Updating image description")

		err = updateImage(fileName, cr.Choices[0].Content)
		if err != nil {
			return fmt.Errorf("update image: %w", err)
		}

		// -------------------------------------------------------------------------

		fmt.Println("  - Generate embeddings for the image description")

		vectors, err := llmEmbed.CreateEmbedding(ctx, []string{cr.Choices[0].Content})
		if err != nil {
			return fmt.Errorf("create embedding: %w", err)
		}

		// -------------------------------------------------------------------------

		fmt.Println("  - Inserting image description into the database")

		d1 := document{
			FileName:    fileName,
			Description: cr.Choices[0].Content,
			Embedding:   vectors[0],
		}

		res, err := col.InsertOne(ctx, d1)
		if err != nil {
			return fmt.Errorf("insert: %w", err)
		}

		fmt.Printf("  - Inserted db id: %s\n", res.InsertedID)
	}

	// We need to give mongodb some time to index the documents.
	// There is no way to know when this gets done.
	time.Sleep(time.Second)

	fmt.Print("\nAsk questions about images (use 'ctrl-c' to quit)\n\n")

	for {
		// -------------------------------------------------------------------------
		// Continue to the LLM search/query part after processing all files

		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Question: ")

		question, _ := reader.ReadString('\n')
		if question == "" {
			return nil
		}

		fmt.Print("\nSearching...\n\n")

		// -------------------------------------------------------------------------

		ctx, cancel := context.WithTimeout(ctx, 240*time.Second)
		defer cancel()

		results, err := vectorSearch(ctx, llmEmbed, col, question)
		if err != nil {
			return fmt.Errorf("vectorSearch: %w", err)
		}

		if err := questionResponse(ctx, llm, question, results); err != nil {
			return fmt.Errorf("questionResponse: %w", err)
		}
	}
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

func getFilesFromDirectory(directoryPath string) ([]string, error) {
	var files []string

	err := filepath.Walk(directoryPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && (filepath.Ext(info.Name()) == ".jpg" || filepath.Ext(info.Name()) == ".jpeg" || filepath.Ext(info.Name()) == ".png") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk directory: %w", err)
	}

	return files, nil
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

func vectorSearch(ctx context.Context, llm *ollama.LLM, col *mongo.Collection, question string) ([]searchResult, error) {

	// -------------------------------------------------------------------------
	// Get the vector embedding for the question.

	embedding, err := llm.CreateEmbedding(ctx, []string{question})
	if err != nil {
		return nil, fmt.Errorf("create embedding: %w", err)
	}

	// -------------------------------------------------------------------------
	// We want to find the nearest neighbors from the question vector embedding.

	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":         "vector_index",
				"exact":         false,
				"path":          "embedding",
				"queryVector":   embedding[0],
				"numCandidates": 5,
				"limit":         5,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"file_name":   1,
				"description": 1,
				"embedding":   1,
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

	// -------------------------------------------------------------------------
	// Return and display the results.

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}

func questionResponse(ctx context.Context, llm *ollama.LLM, question string, results []searchResult) error {
	type searchResult struct {
		FileName    string `json:"file_name"`
		Description string `json:"image_description"`
	}

	var finalResults []searchResult

	fmt.Print("Data:\n\n")
	for _, result := range results {
		if result.Score >= 0.75 {
			fmt.Printf("FileName[%s] Score[%.2f]\n", result.FileName, result.Score)
			finalResults = append(finalResults, searchResult{
				FileName:    result.FileName,
				Description: result.Description,
			})
		}
	}

	content, err := json.Marshal(finalResults)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	// -------------------------------------------------------------------------

	prompt := `
	INSTRUCTIONS:
	
	- Use the following RESULTS to answer the user's question.

	- The data will be a JSON array with the following fields:
	
	[
		{
			"file_name":string,
			"image_description":string
		},
		{
			"file_name":string,
			"image_description":string
		}
	]

	- The response should be in a JSON array with the following fields:
	
	[
		{
			"status": string,
			"filename": string,
			"description": string
		},
		{
			"status": string,
			"filename": string,
			"description": string
		}
	]

	- If there are no RESULTS, provide this response:
	
	[
		{
			"status": "not found"
		}
	]

	- Do not change anything related to the file_name provided.
	- Only provide a brief description of the image.
	- Only provide a valid JSON response.

	RESULTS:
	
	%s
		
	QUESTION:
	
	%s
	`

	finalPrompt := fmt.Sprintf(prompt, string(content), question)

	// This function will display the response as it comes from the server.
	f := func(ctx context.Context, chunk []byte) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		fmt.Printf("%s", chunk)
		return nil
	}

	fmt.Print("\nResults:\n\n")

	// Send the prompt to the model server.
	_, err = llm.Call(
		ctx,
		finalPrompt,
		llms.WithStreamingFunc(f),
		llms.WithMaxTokens(500),
		llms.WithTemperature(0.1),
	)
	if err != nil {
		return fmt.Errorf("call: %w", err)
	}

	fmt.Printf("\n\n")

	return nil
}
