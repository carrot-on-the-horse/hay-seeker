// Synthetic route fixture for CAST approval testing.
package server

type Handler func(string) string

type Router struct {
	routes map[string]Handler
}

func NewRouter() *Router {
	return &Router{routes: make(map[string]Handler)}
}

func (router *Router) Register(method string, path string, handler Handler) {
	router.routes[method+" "+path] = handler
}

func RegisterAPIRoutes(router *Router) {
	router.Register("GET", "/health", health)
	router.Register("POST", "/generate", generate)
}

func health(_ string) string {
	return "ok"
}

func generate(prompt string) string {
	if prompt == "" {
		return "missing prompt"
	}
	return "generated: " + prompt
}
