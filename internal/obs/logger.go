package obs

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/cloudwego/eino/callbacks"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

func New(level string) *slog.Logger {
	var lv slog.Level
	switch strings.ToLower(level) {
	case "debug":
		lv = slog.LevelDebug
	case "warn":
		lv = slog.LevelWarn
	case "error":
		lv = slog.LevelError
	default:
		lv = slog.LevelInfo
	}
	h := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lv})
	return slog.New(h)
}

type ctxKey string

const (
	requestIDKey       ctxKey = "request_id"
	workflowRunIDKey   ctxKey = "workflow_run_id"
	callbackEnabledKey ctxKey = "callback_enabled"
)

var callbackHandlerRegistered atomic.Bool

func InitTracing(_ bool) func(context.Context) error {
	_ = otel.Tracer("agentragplus")
	return func(context.Context) error { return nil }
}

func EnsureRequestID(ctx context.Context) context.Context {
	if v := RequestIDFromContext(ctx); v != "" {
		return ctx
	}
	return context.WithValue(ctx, requestIDKey, "req_"+uuid.NewString())
}

func RequestIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(requestIDKey).(string)
	return strings.TrimSpace(v)
}

func EnsureWorkflowRunID(ctx context.Context) context.Context {
	if v := WorkflowRunIDFromContext(ctx); v != "" {
		return ctx
	}
	return context.WithValue(ctx, workflowRunIDKey, "wf_"+uuid.NewString())
}

func WorkflowRunIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(workflowRunIDKey).(string)
	return strings.TrimSpace(v)
}

func StartSpan(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	ctx = EnsureRequestID(ctx)
	tr := otel.Tracer("agentragplus")
	ctx, span := tr.Start(ctx, name)
	if rid := RequestIDFromContext(ctx); rid != "" {
		attrs = append(attrs, attribute.String("request.id", rid))
	}
	if wf := WorkflowRunIDFromContext(ctx); wf != "" {
		attrs = append(attrs, attribute.String("workflow.run_id", wf))
	}
	if len(attrs) > 0 {
		span.SetAttributes(attrs...)
	}
	return ctx, span
}

func EndSpan(span trace.Span, err error) {
	if span == nil {
		return
	}
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	} else {
		span.SetStatus(codes.Ok, "ok")
	}
	span.End()
}

func EmitEvent(ctx context.Context, name string, attrs ...attribute.KeyValue) {
	span := trace.SpanFromContext(ctx)
	if span == nil {
		return
	}
	span.AddEvent(name, trace.WithAttributes(attrs...))
}

func TraceFields(ctx context.Context) []any {
	out := make([]any, 0, 8)
	if rid := RequestIDFromContext(ctx); rid != "" {
		out = append(out, slog.String("request_id", rid))
	}
	if wf := WorkflowRunIDFromContext(ctx); wf != "" {
		out = append(out, slog.String("workflow_run_id", wf))
	}
	sc := trace.SpanContextFromContext(ctx)
	if sc.IsValid() {
		out = append(out, slog.String("trace_id", sc.TraceID().String()))
		out = append(out, slog.String("span_id", sc.SpanID().String()))
	}
	return out
}

func RegisterGlobalCallbackHandlerIfEnabled(enabled bool, logger *slog.Logger) {
	if !enabled {
		return
	}
	if callbackHandlerRegistered.Swap(true) {
		return
	}
	h := callbacks.NewHandlerBuilder().
		OnStartFn(func(ctx context.Context, info *callbacks.RunInfo, _ callbacks.CallbackInput) context.Context {
			ctx = EnsureRequestID(ctx)
			ctx = EnsureWorkflowRunID(ctx)
			if logger != nil && info != nil {
				logger.Debug("callback_start", slog.String("component", string(info.Component)), slog.String("name", info.Name), slog.String("request_id", RequestIDFromContext(ctx)), slog.String("workflow_run_id", WorkflowRunIDFromContext(ctx)))
			}
			return context.WithValue(ctx, callbackEnabledKey, time.Now())
		}).
		OnEndFn(func(ctx context.Context, info *callbacks.RunInfo, _ callbacks.CallbackOutput) context.Context {
			if logger != nil && info != nil {
				logger.Debug("callback_end", slog.String("component", string(info.Component)), slog.String("name", info.Name), slog.String("request_id", RequestIDFromContext(ctx)), slog.String("workflow_run_id", WorkflowRunIDFromContext(ctx)))
			}
			return ctx
		}).
		OnErrorFn(func(ctx context.Context, info *callbacks.RunInfo, err error) context.Context {
			if logger != nil {
				comp := ""
				name := ""
				if info != nil {
					comp = string(info.Component)
					name = info.Name
				}
				logger.Error("callback_error", slog.String("component", comp), slog.String("name", name), slog.String("error", err.Error()), slog.String("request_id", RequestIDFromContext(ctx)), slog.String("workflow_run_id", WorkflowRunIDFromContext(ctx)))
			}
			return ctx
		}).
		Build()
	callbacks.AppendGlobalHandlers(h)
}
