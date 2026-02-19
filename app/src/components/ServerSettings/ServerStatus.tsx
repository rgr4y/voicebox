import { useQuery } from '@tanstack/react-query';
import { Loader2, RefreshCw, Terminal, XCircle } from 'lucide-react';
import { useState } from 'react';
import { ServerLogViewer } from '@/components/ServerLogViewer/ServerLogViewer';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { apiClient } from '@/lib/api/client';
import { useServerHealth } from '@/lib/hooks/useServer';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { ModelProgress } from './ModelProgress';

export function ServerStatus() {
  const { data: health, isLoading, error } = useServerHealth();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const platform = usePlatform();
  const [restarting, setRestarting] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);

  const { data: modelStatusData } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => apiClient.getModelStatus(),
    refetchInterval: 15_000,
  });

  const models = modelStatusData?.models ?? [];

  const handleRestart = async () => {
    setRestarting(true);
    try {
      const url = await platform.lifecycle.restartServer();
      useServerStore.getState().setServerUrl(url);
    } catch (err) {
      console.error('Failed to restart server:', err);
    } finally {
      setRestarting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Server Status</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="text-sm text-muted-foreground mb-1">Server URL</div>
          <div className="font-mono text-sm">{serverUrl}</div>
        </div>

        {/* Model download progress */}
        <div className="space-y-2">
          {models.map((m) => (
            <ModelProgress
              key={m.model_name}
              modelName={m.model_name}
              displayName={m.display_name}
            />
          ))}
        </div>

        {isLoading ? (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Checking connection...</span>
          </div>
        ) : error ? (
          <div className="flex items-center gap-2">
            <XCircle className="h-4 w-4 text-destructive" />
            <span className="text-sm text-destructive">Connection failed: {error.message}</span>
          </div>
        ) : health ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm">Connected</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge
                variant={health.model_loaded || health.model_downloaded ? 'default' : 'secondary'}
              >
                {health.model_loaded || health.model_downloaded ? 'Model Ready' : 'No Model'}
              </Badge>
              <Badge variant={health.gpu_available ? 'default' : 'secondary'}>
                GPU: {health.gpu_available ? 'Available' : 'Not Available'}
              </Badge>
              {health.vram_used_mb && (
                <Badge variant="outline">VRAM: {health.vram_used_mb.toFixed(0)} MB</Badge>
              )}
            </div>
          </div>
        ) : null}

        {platform.metadata.isTauri && (
          <div className="flex gap-2 pt-1">
            <Button variant="outline" size="sm" disabled={restarting} onClick={handleRestart}>
              {restarting ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4 mr-2" />
              )}
              {restarting ? 'Restarting...' : 'Restart Server'}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setLogsOpen(true)}>
              <Terminal className="h-4 w-4 mr-2" />
              View Server Logs
            </Button>
          </div>
        )}
      </CardContent>
      <ServerLogViewer open={logsOpen} onOpenChange={setLogsOpen} />
    </Card>
  );
}
