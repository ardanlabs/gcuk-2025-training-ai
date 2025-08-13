package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const (
	url   = "http://localhost:11434/v1/chat/completions"
	model = "gpt-oss:latest"
)

// The context window represents the maximum number of tokens that can be sent
// and received by the model. The default for Ollama is 8K. In the makefile
// it has been increased to 64K.
var contextWindow = 1024 * 8

func init() {
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" {
		var err error
		contextWindow, err = strconv.Atoi(v)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := weatherQuestion(context.TODO()); err != nil {
		return fmt.Errorf("weatherQuestion: %w", err)
	}

	return nil
}

func weatherQuestion(ctx context.Context) error {
	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	cln := client.NewSSE[client.ChatSSE](logger)

	// -------------------------------------------------------------------------
	// Start by asking what the weather is like in New York City

	getWeather := NewGetWeather()

	q := "What is the weather like in New York City?"
	fmt.Printf("\nQuestion:\n\n%s\n", q)

	conversation := []client.D{
		{
			"role":    "user",
			"content": q,
		},
	}

	d := client.D{
		"model":       model,
		"messages":    conversation,
		"max_tokens":  contextWindow,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       1,
		"stream":      true,
		"tools": []client.D{
			getWeather.ToolDocument(),
		},
		"tool_selection": "auto",
	}

	ch := make(chan client.ChatSSE, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	// -------------------------------------------------------------------------
	// The model will respond asking us to make the get_current_weather function
	// call. We will make the call and then send the response back to the model.

	fmt.Print("\n")

	for resp := range ch {
		switch {
		case len(resp.Choices[0].Delta.ToolCalls) > 0:
			toolCall := resp.Choices[0].Delta.ToolCalls[0]

			fmt.Printf("\n\nModel Asking For Tool Call:\n\nToolID[%s]: %s(%s)\n\n",
				toolCall.ID,
				toolCall.Function.Name,
				toolCall.Function.Arguments)

			conversation = append(conversation, client.D{
				"role": "assistant",
				"content": fmt.Sprintf("Tool call %s: %s(%v)",
					toolCall.ID,
					toolCall.Function.Name,
					toolCall.Function.Arguments),
			})

			resp := getWeather.Call(ctx, toolCall)
			conversation = append(conversation, resp)

			fmt.Printf("Tool Call Result:\n\n%s\n\n", resp)

		case resp.Choices[0].Delta.Reasoning != "":
			fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
		}
	}

	// -------------------------------------------------------------------------
	// Send the result of the tool call back to the model

	d = client.D{
		"model":       model,
		"messages":    conversation,
		"max_tokens":  contextWindow,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       50,
		"stream":      true,
		"tools": []client.D{
			getWeather.ToolDocument(),
		},
		"tool_selection": "auto",
	}

	ch = make(chan client.ChatSSE, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	// -------------------------------------------------------------------------
	// The model should provide the answer based on the tool call

	fmt.Print("Final Result:\n\n")

	var reasoning bool

	for resp := range ch {
		switch {
		case resp.Choices[0].Delta.Content != "":
			if reasoning {
				fmt.Print("\n\n")
				reasoning = false
			}

			fmt.Print(resp.Choices[0].Delta.Content)

		case resp.Choices[0].Delta.Reasoning != "":
			reasoning = true
			fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
		}
	}

	return nil
}

// =============================================================================

// GetWeather represents a tool that can be used to get the current weather.
type GetWeather struct {
	name string
}

// NewGetWeather creates a new instance of GetWeather.
func NewGetWeather() *GetWeather {
	return &GetWeather{
		name: "tool_get_current_weather",
	}
}

// ToolDocument defines the metadata for the tool that is provied to the model.
func (gw *GetWeather) ToolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        gw.name,
			"description": "Get the current weather for a location",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"location": client.D{
						"type":        "string",
						"description": "The location to get the weather for, e.g. San Francisco, CA",
					},
				},
				"required": []string{"location"},
			},
		},
	}
}

// Call is the function that is called by the agent to get the weather when the
// model requests the tool with the specified parameters.
func (gw *GetWeather) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = client.D{
				"role":         "tool",
				"tool_call_id": toolCall.ID,
				"tool_name":    gw.name,
				"content":      fmt.Sprintf(`{"status": "FAILED", "data": "%s"}`, r),
			}
		}
	}()

	// We are going to hardcode a result for now so we can test the tool.
	// We are going to return the current weather as structured data using JSON
	// which is easier for the model to interpret.

	location := toolCall.Function.Arguments["location"].(string)

	data := map[string]any{
		"temperature": 28,
		"humidity":    80,
		"wind_speed":  10,
		"description": fmt.Sprintln("The weather in", location, "is hot and humid"),
	}

	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: "SUCCESS",
		Data:   data,
	}

	d, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":         "tool",
			"tool_call_id": toolCall.ID,
			"tool_name":    gw.name,
			"content":      fmt.Sprintf(`{"status": "FAILED", "data": "%s"}`, err),
		}
	}

	return client.D{
		"role":         "tool",
		"tool_call_id": toolCall.ID,
		"tool_name":    gw.name,
		"content":      string(d),
	}
}
