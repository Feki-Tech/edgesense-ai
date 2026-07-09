FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 go build -o /edge-agent .

FROM gcr.io/distroless/static-debian12
COPY --from=build /edge-agent /edge-agent
ENTRYPOINT ["/edge-agent"]
