// Synthetic repository fixture. It contains no user or third-party source.
package server

import (
	"encoding/json"
	"net/http"
)

type Router struct {
	routes map[string]http.HandlerFunc
}

func NewRouter() *Router {
	return &Router{routes: make(map[string]http.HandlerFunc)}
}

func (router *Router) GET(path string, handler http.HandlerFunc) {
	router.routes["GET "+path] = handler
}

func (router *Router) POST(path string, handler http.HandlerFunc) {
	router.routes["POST "+path] = handler
}

// RegisterAPIRoutes is the central route-registration function.
func RegisterAPIRoutes(router *Router) {
	router.GET("/api/health", handleHealth)
	router.GET("/api/models", handleModels)
	router.POST("/api/generate", handleGenerate)
	router.POST("/api/chat", handleChat)
}

func handleHealth(response http.ResponseWriter, _ *http.Request) {
	response.WriteHeader(http.StatusNoContent)
}

func handleModels(response http.ResponseWriter, _ *http.Request) {
	models := []string{"tiny", "medium", "large"}
	_ = json.NewEncoder(response).Encode(models)
}

func handleGenerate(response http.ResponseWriter, request *http.Request) {
	var input struct {
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
		http.Error(response, "invalid request", http.StatusBadRequest)
		return
	}
	_ = json.NewEncoder(response).Encode(map[string]string{
		"response": "generated: " + input.Prompt,
	})
}

func handleChat(response http.ResponseWriter, request *http.Request) {
	var input struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
		http.Error(response, "invalid request", http.StatusBadRequest)
		return
	}
	_ = json.NewEncoder(response).Encode(map[string]string{
		"reply": "echo: " + input.Message,
	})
}
