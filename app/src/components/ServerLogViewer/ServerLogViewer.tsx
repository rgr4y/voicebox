import { ArrowDown, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { usePlatform } from '@/platform/PlatformContext';

const MAX_LINES = 2000;

interface ServerLogViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface HttpInfo {
  method: string;
  path: string;
  status: number;
  client?: string;
}

interface LogEntry {
  id: number;
  raw: string;
  // Parsed JSON fields (if valid JSON)
  level?: string;
  message?: string;
  logger?: string;
  ts?: string;
  http?: HttpInfo;
}

let nextId = 0;

function statusColor(status: number): string {
  if (status >= 500) return 'text-red-400';
  if (status >= 400) return 'text-yellow-400';
  if (status >= 300) return 'text-blue-400';
  return 'text-green-400';
}

function parseLine(raw: string): LogEntry {
  const entry: LogEntry = { id: nextId++, raw: raw.trimEnd() };
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null) {
      entry.level = parsed.level || parsed.levelname;
      entry.message = parsed.message;
      entry.logger = parsed.logger || parsed.name;
      entry.ts = parsed.ts || parsed.asctime;

      // uvicorn.access: structured http field alongside message
      if (parsed.http && typeof parsed.http === 'object') {
        entry.http = parsed.http;
      }
    }
  } catch {
    // Not JSON — that's fine, just show raw
  }
  return entry;
}

function levelColor(level?: string): string {
  if (!level) return 'text-muted-foreground';
  switch (level.toUpperCase()) {
    case 'ERROR':
    case 'CRITICAL':
      return 'text-red-400';
    case 'WARNING':
    case 'WARN':
      return 'text-yellow-400';
    case 'DEBUG':
      return 'text-zinc-500';
    default:
      return 'text-muted-foreground';
  }
}

function formatTime(ts?: string): string {
  if (!ts) return '';
  try {
    // Python's logging uses "2026-02-19 12:34:56,789" (comma before ms).
    // Replace the comma with a dot so Date can parse it.
    const normalized = ts.replace(',', '.');
    const d = new Date(normalized);
    if (Number.isNaN(d.getTime())) {
      // If still unparseable, just extract HH:MM:SS from the string
      const match = ts.match(/(\d{2}:\d{2}:\d{2})/);
      return match ? match[1] : '';
    }
    return d.toLocaleTimeString(undefined, {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return '';
  }
}

export function ServerLogViewer({ open, onOpenChange }: ServerLogViewerProps) {
  const platform = usePlatform();
  const [lines, setLines] = useState<LogEntry[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const userScrolledRef = useRef(false);

  // Subscribe to server log events
  useEffect(() => {
    if (!open) return;

    let unlisten: (() => void) | null = null;

    platform.lifecycle
      .onServerLog((line) => {
        const entry = parseLine(line);
        setLines((prev) => {
          const next = [...prev, entry];
          return next.length > MAX_LINES ? next.slice(-MAX_LINES) : next;
        });
      })
      .then((fn) => {
        unlisten = fn;
      });

    return () => {
      unlisten?.();
    };
  }, [open, platform.lifecycle]);

  // Auto-scroll to bottom — `lines` is intentionally a dependency to trigger on new data
  // biome-ignore lint/correctness/useExhaustiveDependencies: lines triggers scroll on new log entries
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  // Detect user scroll to pause auto-scroll
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    if (!atBottom && !userScrolledRef.current) {
      userScrolledRef.current = true;
      setAutoScroll(false);
    } else if (atBottom && userScrolledRef.current) {
      userScrolledRef.current = false;
      setAutoScroll(true);
    }
  }, []);

  const handleClear = () => {
    setLines([]);
  };

  const handleScrollToBottom = () => {
    userScrolledRef.current = false;
    setAutoScroll(true);
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl w-[90vw] h-[70vh] flex flex-col p-0 gap-0">
        <DialogHeader className="px-4 pt-4 pb-2 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="text-sm font-medium">Server Logs</DialogTitle>
              <DialogDescription className="text-xs">
                {lines.length} line{lines.length !== 1 ? 's' : ''} • streaming from sidecar
              </DialogDescription>
            </div>
            <div className="flex items-center gap-1 mr-6">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={handleClear}
                title="Clear logs"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
              {!autoScroll && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 p-0"
                  onClick={handleScrollToBottom}
                  title="Scroll to bottom"
                >
                  <ArrowDown className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          </div>
        </DialogHeader>
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto overflow-x-auto bg-zinc-950 border-t border-border"
        >
          <pre className="text-[11px] leading-[1.6] font-mono p-3 min-h-full">
            {lines.length === 0 ? (
              <span className="text-zinc-600 italic">Waiting for server output…</span>
            ) : (
              lines.map((entry) => (
                <div key={entry.id} className="hover:bg-zinc-900/50">
                  {entry.level ? (
                    <>
                      <span className="text-zinc-600">{formatTime(entry.ts)} </span>
                      <span className={levelColor(entry.level)}>
                        {entry.level.toUpperCase().padEnd(8)}
                      </span>
                      <span className="text-zinc-500">
                        {entry.logger ? `[${entry.logger}] ` : ''}
                      </span>
                      {entry.http ? (
                        <span>
                          <span className="text-zinc-400">{entry.http.method} </span>
                          <span className="text-zinc-300">{entry.http.path} </span>
                          <span className={statusColor(entry.http.status)}>
                            {entry.http.status}
                          </span>
                        </span>
                      ) : (
                        <span className="text-zinc-300">{entry.message ?? entry.raw}</span>
                      )}
                    </>
                  ) : (
                    <span className="text-zinc-400">{entry.raw}</span>
                  )}
                </div>
              ))
            )}
          </pre>
        </div>
      </DialogContent>
    </Dialog>
  );
}
