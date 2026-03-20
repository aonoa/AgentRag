APP=agentragplus

.PHONY: help run-default run-local run-dev run-staging run-prod test build fmt

help:
	@printf "Targets:\n"
	@printf "  run-default  Run server with APP_ENV=default\n"
	@printf "  run-local    Run server with APP_ENV=local\n"
	@printf "  run-dev      Run server with APP_ENV=dev\n"
	@printf "  run-staging  Run server with APP_ENV=staging\n"
	@printf "  run-prod     Run server with APP_ENV=prod\n"
	@printf "  test         Run all tests\n"
	@printf "  build        Build all packages\n"
	@printf "  fmt          Format Go files\n"

run-default:
	APP_ENV=default go run ./cmd/server

run-local:
	APP_ENV=local go run ./cmd/server

run-dev:
	APP_ENV=dev go run ./cmd/server

run-staging:
	APP_ENV=staging go run ./cmd/server

run-prod:
	APP_ENV=prod go run ./cmd/server

test:
	go test ./...

build:
	go build ./...

fmt:
	gofmt -w $$(find . -name '*.go' -type f)
