import { zodResolver } from '@hookform/resolvers/zod';
import { useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { LANGUAGE_CODES, type LanguageCode } from '@/lib/constants/languages';
import { useGeneration } from '@/lib/hooks/useGeneration';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

const usePlayerReset = () => usePlayerStore((state) => state.reset);

const generationSchema = z.object({
  text: z.string().min(1, 'Text is required').max(5000),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  seed: z.number().int().optional(),
  modelSize: z.string().optional(),
  instruct: z.string().max(500).optional(),
});

export type GenerationFormValues = z.infer<typeof generationSchema>;

interface UseGenerationFormOptions {
  onSuccess?: (generationId: string) => void;
  defaultValues?: Partial<GenerationFormValues>;
}

export function useGenerationForm(options: UseGenerationFormOptions = {}) {
  const { toast } = useToast();
  const generation = useGeneration();
  const setAudioWithAutoPlay = usePlayerStore((state) => state.setAudioWithAutoPlay);
  const resetPlayer = usePlayerReset();
  const setIsGenerating = useGenerationStore((state) => state.setIsGenerating);

  // Fetch model status so we can validate/correct the selected model size
  const { data: modelStatusData } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => apiClient.getModelStatus(),
    refetchInterval: 15_000,
  });

  const ttsModels = (modelStatusData?.models ?? []).filter((m) => m.model_type === 'tts');
  const installedTtsModels = ttsModels.filter((m) => m.downloaded);
  const installedSizes = installedTtsModels.map((m) => m.model_size).filter(Boolean) as string[];
  const noModelsInstalled = modelStatusData !== undefined && installedTtsModels.length === 0;

  const form = useForm<GenerationFormValues>({
    resolver: zodResolver(generationSchema),
    defaultValues: {
      text: '',
      language: 'en',
      seed: undefined,
      modelSize: localStorage.getItem('voicebox:lastModelSize') ?? undefined,
      instruct: '',
      ...options.defaultValues,
    },
  });

  // Correct model size: keep current if installed, else pick first installed
  useEffect(() => {
    if (installedSizes.length === 0) return;
    const current = form.getValues('modelSize');
    if (!current || !installedSizes.includes(current)) {
      form.setValue('modelSize', installedSizes[0]);
    }
  }, [installedSizes.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleSubmit(
    data: GenerationFormValues,
    selectedProfileId: string | null,
  ): Promise<void> {
    if (!selectedProfileId) {
      toast({
        title: 'No profile selected',
        description: 'Please select a voice profile from the cards above.',
        variant: 'destructive',
      });
      return;
    }

    try {
      setIsGenerating(true);
      resetPlayer(); // Close any existing audio player

      if (data.modelSize) {
        localStorage.setItem('voicebox:lastModelSize', data.modelSize);
      }

      const result = await generation.mutateAsync({
        profile_id: selectedProfileId,
        text: data.text,
        language: data.language,
        seed: data.seed,
        model_size: data.modelSize,
        instruct: data.instruct || undefined,
      });

      toast({
        title: 'Generation complete!',
        description: `Audio generated (${result.duration.toFixed(2)}s)`,
      });

      const audioUrl = apiClient.getAudioUrl(result.id);
      setAudioWithAutoPlay(audioUrl, result.id, selectedProfileId, data.text.substring(0, 50));

      // Preserve sticky fields across reset — only clear text/seed/instruct
      form.reset({
        text: '',
        language: form.getValues('language'),
        seed: undefined,
        modelSize: form.getValues('modelSize'),
        instruct: '',
      });
      options.onSuccess?.(result.id);
    } catch (error) {
      toast({
        title: 'Generation failed',
        description: error instanceof Error ? error.message : 'Failed to generate audio',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
    }
  }

  const pendingJobs = useGenerationStore((state) => state.pendingJobs);
  const isQueueLimitReached = pendingJobs.length >= 3;

  return {
    form,
    handleSubmit,
    isPending: generation.isPending,
    isQueueLimitReached,
    ttsModels,
    installedSizes,
    noModelsInstalled,
    modelLocked: false,
  };
}
