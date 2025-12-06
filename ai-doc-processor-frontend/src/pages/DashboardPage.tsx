import { useState, useRef } from 'react';
import {
    Container,
    Title,
    Text,
    Stack,
    Grid,
    Card,
    Group,
    Button,
    Tabs,
    Alert,
    Divider,
    ThemeIcon,
    Switch,
    Badge,
    Timeline,
    Progress,
    RingProgress,
    Loader,
    Modal,
    Textarea,
    ActionIcon,
    Paper,
} from '@mantine/core';
import { IconInfoCircle, IconUpload, IconFileAnalytics, IconCheck, IconRefresh, IconClock, IconFileStack, IconLoader, IconChartBar, IconMessageCircle, IconStar, IconEdit, IconServer, IconWifi, IconWifiOff } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';

import { FileDropZone } from '@/components/FileUpload/DropZone';
import { FileCard } from '@/components/FileUpload/FileCard';
import { StatusCard } from '@/components/Processing/StatusCard';
import type { ProcessingStatus } from '@/components/Processing/StatusCard';
import { useBackendStatus } from '@/hooks/useBackendStatus';

import { DataTable } from '@/components/Results/DataTable';
import { JSONViewer } from '@/components/Results/JSONViewer';
import { DownloadButtons } from '@/components/Results/DownloadButtons';
import { processingService } from '@/services';
import type { ProcessingResult, ChunkedProcessingResult, ChunkInfo, ChunkProgressEvent } from '@/services';

// Live chunk progress state
interface LiveChunkStatus {
    chunk: number;
    totalChunks: number;
    pageRange?: string;
    sectionType?: string;
    header?: string;
    status: 'pending' | 'started' | 'extracting' | 'processing' | 'processing_with_context' | 'success' | 'error';
    entriesCount?: number;
    processingTime?: number;
    error?: string;
}

export function DashboardPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [processingStatus, setProcessingStatus] = useState<ProcessingStatus>('idle');
    const [result, setResult] = useState<ProcessingResult | null>(null);
    
    // Backend status monitoring
    const { status: backendStatus, health, isOnline, refresh: refreshBackend } = useBackendStatus();
    const [rawJSON, setRawJSON] = useState<string>('');
    const [error, setError] = useState<string | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);
    const [processingMode, setProcessingMode] = useState<'character'>('character');
    const [chunkInfo, setChunkInfo] = useState<ChunkedProcessingResult | null>(null);
    const notificationTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Analytics and Feedback state
    const [showAnalytics, setShowAnalytics] = useState(false);
    const [showFeedback, setShowFeedback] = useState(false);
    const [analytics, setAnalytics] = useState<any>(null);
    const [loadingAnalytics, setLoadingAnalytics] = useState(false);
    const [feedbackRating, setFeedbackRating] = useState(0);
    const [feedbackText, setFeedbackText] = useState('');
    const [submittingFeedback, setSubmittingFeedback] = useState(false);
    const [autoFeedback, setAutoFeedback] = useState<any>(null);
    const [loadingAutoFeedback, setLoadingAutoFeedback] = useState(false);
    const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
    const [cardDismissed, setCardDismissed] = useState(false);

    // Live streaming progress state
    const [liveProgress, setLiveProgress] = useState<{
        isStreaming: boolean;
        totalChunks: number;
        totalPages: number;
        currentChunk: number;
        chunks: LiveChunkStatus[];
        elapsedTime: number;
        analysisPhase: 'idle' | 'analyzing' | 'complete';
        mergingPhase: 'idle' | 'merging' | 'complete';
        sectionsFound: number;
        strategy: string;
    }>({
        isStreaming: false,
        totalChunks: 0,
        totalPages: 0,
        currentChunk: 0,
        chunks: [],
        elapsedTime: 0,
        analysisPhase: 'idle',
        mergingPhase: 'idle',
        sectionsFound: 0,
        strategy: '',
    });
    const elapsedTimerRef = useRef<NodeJS.Timeout | null>(null);

    const handleFileSelect = (file: File) => {
        setSelectedFile(file);
        setResult(null);
        setError(null);
        setProcessingStatus('idle');

        // Clear any pending notification timeout
        if (notificationTimeoutRef.current) {
            clearTimeout(notificationTimeoutRef.current);
        }

        // Hide any existing notification immediately
        notifications.hide('file-selected');

        // Debounce the notification to prevent duplicates
        notificationTimeoutRef.current = setTimeout(() => {
            notifications.show({
                id: 'file-selected',
                title: 'File Selected',
                message: `${file.name} is ready to process`,
                color: 'violet',
                autoClose: 3000,
            });
        }, 100);
    };

    const handleRemoveFile = () => {
        if (notificationTimeoutRef.current) {
            clearTimeout(notificationTimeoutRef.current);
        }
        notifications.hide('file-selected');
        setSelectedFile(null);
        setResult(null);
        setError(null);
        setProcessingStatus('idle');
    };

    const handleProcess = async () => {
        if (!selectedFile) return;

        setError(null);
        setChunkInfo(null);
        setProcessingStatus('extracting');

        try {
            // Character-based processing (default and only mode)
            setProcessingStatus('analyzing');

            // Reset live progress for character processing
            setLiveProgress({
                isStreaming: true,
                totalChunks: 0,
                totalPages: 0,
                currentChunk: 0,
                chunks: [],
                elapsedTime: 0,
                analysisPhase: 'idle',
                mergingPhase: 'idle',
                sectionsFound: 0,
                strategy: '',
            });

            // Start elapsed time counter
            const startTime = Date.now();
            elapsedTimerRef.current = setInterval(() => {
                setLiveProgress(prev => ({
                    ...prev,
                    elapsedTime: Math.round((Date.now() - startTime) / 1000),
                }));
            }, 1000);

            const handleCharacterProgress = (event: ChunkProgressEvent) => {
                if (event.event === 'analysis_start') {
                    setLiveProgress(prev => ({ ...prev, analysisPhase: 'analyzing' }));
                } else if (event.event === 'analysis_complete') {
                    const initialChunks: LiveChunkStatus[] = [];
                    for (let i = 1; i <= (event.total_chunks || 0); i++) {
                        initialChunks.push({
                            chunk: i,
                            totalChunks: event.total_chunks || 0,
                            status: 'pending',
                        });
                    }
                    setLiveProgress(prev => ({
                        ...prev,
                        analysisPhase: 'complete',
                        totalChunks: event.total_chunks || 0,
                        strategy: `${event.strategy} (${event.chunk_size} chars, ${event.overlap} overlap)`,
                        chunks: initialChunks,
                    }));
                } else if (event.event === 'chunk_start') {
                    setLiveProgress(prev => ({
                        ...prev,
                        currentChunk: event.chunk || 0,
                        chunks: prev.chunks.map(c =>
                            c.chunk === event.chunk
                                ? {
                                    ...c,
                                    status: 'started',
                                    pageRange: event.page_range,
                                    sectionType: event.char_range
                                }
                                : c
                        ),
                    }));
                } else if (event.event === 'chunk_progress') {
                    setLiveProgress(prev => ({
                        ...prev,
                        chunks: prev.chunks.map(c =>
                            c.chunk === event.chunk
                                ? { ...c, status: event.status as LiveChunkStatus['status'] }
                                : c
                        ),
                    }));
                } else if (event.event === 'chunk_complete') {
                    setLiveProgress(prev => ({
                        ...prev,
                        chunks: prev.chunks.map(c =>
                            c.chunk === event.chunk
                                ? {
                                    ...c,
                                    status: event.status === 'success' ? 'success' : 'error',
                                    entriesCount: event.entries_count,
                                    processingTime: event.processing_time,
                                    error: event.error,
                                }
                                : c
                        ),
                    }));
                } else if (event.event === 'merging_start') {
                    setLiveProgress(prev => ({ ...prev, mergingPhase: 'merging' }));
                }
            };

            const characterResult = await processingService.processPDF(
                selectedFile,
                handleCharacterProgress
            );

            // Stop elapsed timer
            if (elapsedTimerRef.current) {
                clearInterval(elapsedTimerRef.current);
            }

            setLiveProgress(prev => ({ ...prev, isStreaming: false, mergingPhase: 'complete' }));
            setChunkInfo(characterResult);
            setResult({
                entries: characterResult.entries,
                global_notes: characterResult.global_notes,
                session_id: characterResult.session_id,
                processing_time: characterResult.processing_time
            });
            setRawJSON(JSON.stringify(characterResult, null, 2));
            setProcessingStatus('complete');
            setFeedbackSubmitted(false);
            setCardDismissed(false);

            // Fetch auto-feedback automatically with retry logic
            if (characterResult.session_id) {
                const fetchAutoFeedback = async (retries = 5) => {
                    for (let i = 0; i < retries; i++) {
                        try {
                            await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1))); // Wait 2s, 4s, 6s, 8s, 10s
                            console.log(`🔄 Fetching auto-feedback (attempt ${i + 1}/${retries})...`);
                            const response = await fetch(`${import.meta.env.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:8001'}/api/feedback/${characterResult.session_id}`);
                            const data = await response.json();
                            console.log(`📥 Response:`, data);
                            if (data.success && data.feedback) {
                                setAutoFeedback(data.feedback);
                                console.log('✅ Auto-feedback loaded:', data.feedback);
                                return;
                            }
                        } catch (err) {
                            console.error(`❌ Attempt ${i + 1} failed:`, err);
                        }
                    }
                    console.error('❌ Failed to fetch auto-feedback after 5 attempts');
                    console.log('💡 Session ID:', characterResult.session_id);
                    // Set a placeholder so card shows manual feedback option
                    setAutoFeedback({
                        rating: 0,
                        feedback_text: 'Auto-feedback unavailable. Please provide manual feedback.',
                        feedback_id: 'manual_fallback'
                    });
                };
                fetchAutoFeedback();
            }

            notifications.show({
                id: 'processing-complete',
                title: 'Processing Complete!',
                message: `Processed ${characterResult.total_chunks} character chunks in ${characterResult.processing_time}s, extracted ${characterResult.entries.length} rows`,
                color: 'teal',
                autoClose: 5000,
            });
        } catch (err: any) {
            // Clean up timer on error
            if (elapsedTimerRef.current) {
                clearInterval(elapsedTimerRef.current);
            }
            setLiveProgress(prev => ({ ...prev, isStreaming: false }));

            const errorMessage = err.message || 'Processing failed';
            setError(errorMessage);
            setProcessingStatus('error');

            notifications.show({
                id: 'processing-error',
                title: 'Processing Failed',
                message: errorMessage,
                color: 'red',
                autoClose: 8000,
            });
        }
    };

    const handleDownloadExcel = async () => {
        if (!result || isDownloading) return;

        setIsDownloading(true);
        try {
            await processingService.downloadExcel(result, `${selectedFile?.name.replace('.pdf', '')}_extracted.xlsx`);
            notifications.show({
                id: 'download-excel',
                title: 'Download Started',
                message: 'Excel file is being downloaded',
                color: 'teal',
                autoClose: 3000,
            });
        } catch (err: any) {
            notifications.show({
                id: 'download-error',
                title: 'Download Failed',
                message: err.message || 'Failed to download Excel file',
                color: 'red',
                autoClose: 5000,
            });
        } finally {
            setIsDownloading(false);
        }
    };

    const handleDownloadJSON = () => {
        if (!result || isDownloading) return;

        setIsDownloading(true);
        try {
            processingService.downloadJSON(result, `${selectedFile?.name.replace('.pdf', '')}_extracted.json`);
            notifications.show({
                id: 'download-json',
                title: 'Download Started',
                message: 'JSON file is being downloaded',
                color: 'teal',
                autoClose: 3000,
            });
        } catch (err: any) {
            notifications.show({
                id: 'download-error',
                title: 'Download Failed',
                message: err.message || 'Failed to download JSON file',
                color: 'red',
                autoClose: 5000,
            });
        } finally {
            setIsDownloading(false);
        }
    };

    const handleFetchAnalytics = async () => {
        setLoadingAnalytics(true);
        try {
            const response = await fetch(`${import.meta.env.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:8001'}/api/analytics`);
            const data = await response.json();
            setAnalytics(data);
            setShowAnalytics(true);

            notifications.show({
                title: 'Analytics Loaded',
                message: 'System analytics retrieved successfully',
                color: 'blue',
                autoClose: 3000,
            });
        } catch (err: any) {
            notifications.show({
                title: 'Analytics Error',
                message: err.message || 'Failed to fetch analytics',
                color: 'red',
                autoClose: 5000,
            });
        } finally {
            setLoadingAnalytics(false);
        }
    };

    const handleOpenFeedback = async () => {
        // Reset rating to 0 when opening modal
        setFeedbackRating(0);
        setFeedbackText('');
        setShowFeedback(true);

        // Fetch auto-generated feedback if session exists
        if (result?.session_id && !autoFeedback) {
            setLoadingAutoFeedback(true);
            try {
                const response = await fetch(`${import.meta.env.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:8001'}/api/feedback/${result.session_id}`);
                const data = await response.json();

                if (data.success && data.feedback) {
                    setAutoFeedback(data.feedback);
                    // Don't pre-fill rating - let user choose
                }
            } catch (err) {
                console.error('Failed to fetch auto-feedback:', err);
            } finally {
                setLoadingAutoFeedback(false);
            }
        }
    };

    const handleAgreeWithRating = async () => {
        if (!result?.session_id || !autoFeedback) return;

        setSubmittingFeedback(true);
        try {
            // Submit agreement as feedback with same rating
            const response = await fetch(`${import.meta.env.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:8001'}/api/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: result.session_id,
                    rating: autoFeedback.rating,
                    feedback_text: 'User agreed with automated rating',
                    corrections: [],
                }),
            });

            const data = await response.json();

            if (data.success) {
                setFeedbackSubmitted(true);
                setCardDismissed(true);
                notifications.show({
                    title: 'Thank You! 🎉',
                    message: 'Your agreement helps validate our system. Keep up the great work!',
                    color: 'green',
                    autoClose: 4000,
                });
            } else {
                throw new Error(data.message || 'Failed to submit agreement');
            }
        } catch (err: any) {
            notifications.show({
                title: 'Submission Error',
                message: err.message || 'Failed to submit agreement',
                color: 'red',
                autoClose: 5000,
            });
        } finally {
            setSubmittingFeedback(false);
        }
    };

    const handleSubmitFeedback = async () => {
        if (feedbackRating === 0) {
            notifications.show({
                title: 'Rating Required',
                message: 'Please select a rating',
                color: 'orange',
                autoClose: 3000,
            });
            return;
        }

        // Use the most recent session_id or a test session
        const sessionId = result?.session_id || 'test_session_general';

        setSubmittingFeedback(true);
        try {
            const response = await fetch(`${import.meta.env.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:8001'}/api/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    rating: feedbackRating,
                    feedback_text: feedbackText,
                    corrections: [],
                }),
            });

            const data = await response.json();

            if (data.success) {
                const isOverride = autoFeedback && autoFeedback.rating !== feedbackRating;
                setFeedbackSubmitted(true);
                setCardDismissed(true);
                notifications.show({
                    title: isOverride ? 'Feedback Override Submitted 🎉' : 'Feedback Submitted 🎉',
                    message: isOverride ? 'Your rating has been updated! This helps improve the system.' : 'Thank you for your feedback! This helps improve the system.',
                    color: 'green',
                    autoClose: 4000,
                });
                setShowFeedback(false);
                setFeedbackRating(0);
                setFeedbackText('');
            } else {
                throw new Error(data.message || 'Failed to submit feedback');
            }
        } catch (err: any) {
            notifications.show({
                title: 'Feedback Error',
                message: err.message || 'Failed to submit feedback',
                color: 'red',
                autoClose: 5000,
            });
        } finally {
            setSubmittingFeedback(false);
        }
    };

    return (
        <Container size="xl" py="xl">
            <Stack gap="xl">
                {/* Header */}
                <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                    <Title order={1} mb="xs" style={{
                        background: 'linear-gradient(45deg, #7b31e6, #5f23b6)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        fontSize: '3rem'
                    }}>
                        AI Document Processor
                    </Title>
                    <Text c="dimmed" size="xl" maw={600} mx="auto" mb="md">
                        Transform unstructured PDFs into structured Excel files with <span style={{ color: '#7b31e6', fontWeight: 600 }}>100% fidelity</span>
                    </Text>

                    {/* Analytics and Feedback Buttons */}
                    <Group justify="center" gap="md" mt="lg">
                        <Button
                            leftSection={<IconChartBar size={18} />}
                            variant="light"
                            color="blue"
                            onClick={handleFetchAnalytics}
                            loading={loadingAnalytics}
                        >
                            View Analytics
                        </Button>
                        <Button
                            leftSection={<IconMessageCircle size={18} />}
                            variant="light"
                            color={result?.session_id ? 'violet' : 'gray'}
                            onClick={handleOpenFeedback}
                            disabled={!result?.session_id && processingStatus !== 'complete'}
                        >
                            {result?.session_id ? 'Override Auto-Feedback' : 'Submit Feedback'}
                        </Button>
                    </Group>
                </div>

                {/* Info Alert */}
                <Alert
                    icon={<IconInfoCircle />}
                    title="How It Works"
                    color="violet"
                    variant="light"
                    radius="md"
                    styles={{
                        root: { backdropFilter: 'blur(10px)', backgroundColor: 'rgba(246, 240, 255, 0.8)' }
                    }}
                >
                    <Text size="sm">
                        1. Upload your PDF document • 2. AI extracts data dynamically • 3. Review structured output • 4. Download as Excel
                    </Text>
                </Alert>

                <Grid gutter="xl">
                    {/* Left Column - Upload & Processing */}
                    <Grid.Col span={{ base: 12, md: 8 }}>
                        <Stack gap="lg">
                            {/* Upload Section */}
                            <Card shadow="sm" padding="xl" radius="lg" withBorder style={{ overflow: 'visible' }}>
                                <Stack gap="md">
                                    <Group>
                                        <ThemeIcon size="lg" radius="md" variant="light" color="violet">
                                            <IconUpload size={20} />
                                        </ThemeIcon>
                                        <Text size="lg" fw={600}>
                                            Upload Document
                                        </Text>
                                    </Group>

                                    {!selectedFile ? (
                                        <FileDropZone
                                            onFileSelect={handleFileSelect}
                                            disabled={processingStatus !== 'idle' && processingStatus !== 'complete'}
                                        />
                                    ) : (
                                        <FileCard
                                            file={{
                                                name: selectedFile.name,
                                                size: selectedFile.size,
                                            }}
                                            onRemove={handleRemoveFile}
                                        />
                                    )}

                                    {selectedFile && processingStatus === 'idle' && (
                                        <>
                                            <Card padding="md" withBorder style={{ background: 'rgba(59, 130, 246, 0.05)' }}>
                                                <Stack gap="sm">
                                                    <Group>
                                                        <Text size="sm" fw={600} c="blue">📝 Processing Mode</Text>
                                                        <Badge size="sm" color="blue" variant="light">Character-Based</Badge>
                                                    </Group>
                                                    <Text size="xs" c="dimmed">
                                                        Fixed-size chunks (1200 chars) with 100-char overlap for optimal data extraction and consistent performance.
                                                    </Text>
                                                </Stack>
                                            </Card>
                                            <Button
                                                size="lg"
                                                fullWidth
                                                onClick={handleProcess}
                                                leftSection={<IconFileStack size={20} />}
                                                variant="gradient"
                                                gradient={{ from: 'violet', to: 'indigo' }}
                                                style={{ transition: 'transform 0.2s' }}
                                            >
                                                📝 Process Document
                                            </Button>
                                        </>
                                    )}
                                </Stack>
                            </Card>

                            {/* Live Chunk Progress */}
                            {liveProgress.isStreaming && (
                                <Card shadow="sm" padding="lg" radius="lg" withBorder style={{ background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
                                    <Stack gap="md">
                                        <Group justify="space-between">
                                            <Group>
                                                <Loader size="sm" color="blue" />
                                                <Text size="lg" fw={600}>
                                                    📝 Character Processing
                                                </Text>
                                            </Group>
                                            <Group gap="xs">
                                                <Badge color="blue" variant="light" leftSection={<IconClock size={12} />}>
                                                    {liveProgress.elapsedTime}s elapsed
                                                </Badge>

                                                <Badge color="violet" variant="light">
                                                    {liveProgress.chunks.filter(c => c.status === 'success').length}/{liveProgress.totalChunks} chunks
                                                </Badge>
                                            </Group>
                                        </Group>

                                        {/* Character Processing Phases */}
                                        <Stack gap="xs">
                                            <Group gap="sm">
                                                {liveProgress.analysisPhase === 'analyzing' && <Loader size="xs" color="blue" />}
                                                {liveProgress.analysisPhase === 'complete' && <IconCheck size={16} color="green" />}
                                                <Text size="sm" c={liveProgress.analysisPhase === 'complete' ? 'teal' : 'dimmed'}>
                                                    Text Extraction & Analysis
                                                </Text>
                                            </Group>
                                            <Group gap="sm">
                                                {liveProgress.totalChunks > 0 && liveProgress.chunks.filter(c => c.status === 'success').length < liveProgress.totalChunks && <Loader size="xs" color="blue" />}
                                                {liveProgress.chunks.filter(c => c.status === 'success').length === liveProgress.totalChunks && liveProgress.totalChunks > 0 && <IconCheck size={16} color="green" />}
                                                <Text size="sm" c={liveProgress.chunks.filter(c => c.status === 'success').length === liveProgress.totalChunks && liveProgress.totalChunks > 0 ? 'teal' : 'dimmed'}>
                                                    Character Chunk Processing
                                                </Text>
                                            </Group>
                                            <Group gap="sm">
                                                {liveProgress.mergingPhase === 'merging' && <Loader size="xs" color="blue" />}
                                                {liveProgress.mergingPhase === 'complete' && <IconCheck size={16} color="green" />}
                                                <Text size="sm" c={liveProgress.mergingPhase === 'complete' ? 'teal' : 'dimmed'}>
                                                    Merging & Deduplication
                                                </Text>
                                            </Group>
                                        </Stack>

                                        <Progress
                                            value={(liveProgress.chunks.filter(c => c.status === 'success').length / liveProgress.totalChunks) * 100}
                                            color="violet"
                                            size="lg"
                                            radius="xl"
                                            animated
                                        />

                                        <Stack gap="xs">
                                            {liveProgress.chunks.map((chunk) => (
                                                <Group key={chunk.chunk} justify="space-between" p="xs" style={{
                                                    background: chunk.status === 'success' ? 'rgba(16, 185, 129, 0.1)' :
                                                        chunk.status === 'error' ? 'rgba(239, 68, 68, 0.1)' :
                                                            ['started', 'extracting', 'processing', 'processing_with_context'].includes(chunk.status) ? 'rgba(99, 102, 241, 0.1)' :
                                                                'rgba(0, 0, 0, 0.02)',
                                                    borderRadius: '8px',
                                                    transition: 'all 0.3s ease'
                                                }}>
                                                    <Group gap="sm">
                                                        {chunk.status === 'success' && <IconCheck size={16} color="green" />}
                                                        {chunk.status === 'error' && <Text c="red" size="sm">✗</Text>}
                                                        {['started', 'extracting', 'processing', 'processing_with_context'].includes(chunk.status) && <Loader size="xs" color="indigo" />}
                                                        {chunk.status === 'pending' && <Text c="dimmed" size="sm">○</Text>}
                                                        <Stack gap={2}>
                                                            <Text size="sm" fw={500}>
                                                                Chunk {chunk.chunk}/{chunk.totalChunks}
                                                                {chunk.pageRange && <Text span c="dimmed" size="xs"> (Pages {chunk.pageRange})</Text>}
                                                            </Text>
                                                            {chunk.sectionType && (
                                                                <Group gap="xs">
                                                                    <Badge size="xs" color="blue" variant="light">
                                                                        Chars {chunk.sectionType}
                                                                    </Badge>
                                                                </Group>
                                                            )}
                                                        </Stack>
                                                    </Group>
                                                    <Group gap="xs">
                                                        {chunk.status === 'started' && <Badge size="xs" color="blue">Started</Badge>}
                                                        {chunk.status === 'extracting' && <Badge size="xs" color="indigo">Extracting</Badge>}
                                                        {chunk.status === 'processing' && <Badge size="xs" color="violet">Processing with AI</Badge>}
                                                        {chunk.status === 'processing_with_context' && <Badge size="xs" color="grape">AI + Context</Badge>}
                                                        {chunk.status === 'success' && (
                                                            <>
                                                                <Badge size="xs" color="teal">{chunk.entriesCount} entries</Badge>
                                                                <Badge size="xs" color="gray">{chunk.processingTime}s</Badge>
                                                            </>
                                                        )}
                                                        {chunk.status === 'error' && <Badge size="xs" color="red">Error</Badge>}
                                                        {chunk.status === 'pending' && <Badge size="xs" color="gray" variant="light">Pending</Badge>}
                                                    </Group>
                                                </Group>
                                            ))}
                                        </Stack>
                                    </Stack>
                                </Card>
                            )}

                            {/* Processing Status */}
                            {processingStatus !== 'idle' && !liveProgress.isStreaming && (
                                <>
                                    <StatusCard status={processingStatus} error={error || undefined} />

                                    {/* Retry Button on Error */}
                                    {processingStatus === 'error' && selectedFile && (
                                        <Button
                                            size="md"
                                            fullWidth
                                            onClick={handleProcess}
                                            leftSection={<IconRefresh size={18} />}
                                            variant="light"
                                            color="violet"
                                        >
                                            Retry Processing
                                        </Button>
                                    )}
                                </>
                            )}

                            {/* Chunk Information */}
                            {chunkInfo && processingStatus === 'complete' && (
                                <Card shadow="sm" padding="lg" radius="lg" withBorder style={{ background: 'linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%)' }}>
                                    <Stack gap="md">
                                        <Group>
                                            <ThemeIcon size="lg" radius="md" variant="light" color="blue">
                                                <IconFileStack size={20} />
                                            </ThemeIcon>
                                            <div>
                                                <Text size="lg" fw={600}>
                                                    Chunk Processing Details
                                                </Text>
                                                <Group gap="xs" mt={4}>
                                                    <Badge color="blue" variant="light" leftSection={<IconClock size={12} />}>
                                                        {chunkInfo.processing_time}s total
                                                    </Badge>
                                                    <Badge color="violet" variant="light">
                                                        {chunkInfo.total_chunks} chunks
                                                    </Badge>
                                                    <Badge color="teal" variant="light">
                                                        {chunkInfo.total_pages} pages
                                                    </Badge>
                                                </Group>
                                            </div>
                                        </Group>

                                        {(() => {
                                            const totalDuplicates = chunkInfo.chunks.reduce((sum, chunk) =>
                                                sum + ((chunk as any).duplicates_removed || 0), 0
                                            );
                                            return totalDuplicates > 0 && (
                                                <Alert color="orange" variant="light" icon={<IconCheck size={16} />}>
                                                    <Text size="xs">
                                                        Removed {totalDuplicates} duplicate entries across all chunks for cleaner results
                                                    </Text>
                                                </Alert>
                                            );
                                        })()}

                                        <Divider />

                                        <Timeline active={chunkInfo.chunks.length} bulletSize={24} lineWidth={2} color="violet">
                                            {chunkInfo.chunks.map((chunk) => (
                                                <Timeline.Item
                                                    key={chunk.chunk_id}
                                                    bullet={chunk.status === 'success' ? <IconCheck size={12} /> : <Text size="xs">{chunk.chunk_id}</Text>}
                                                    title={
                                                        <Group gap="xs">
                                                            <Text size="sm" fw={500}>
                                                                Chunk {chunk.chunk_id}: Pages {chunk.page_range}
                                                            </Text>
                                                            <Badge size="xs" color={chunk.status === 'success' ? 'teal' : 'red'}>
                                                                {chunk.status}
                                                            </Badge>
                                                        </Group>
                                                    }
                                                >
                                                    <Group gap="md" mt={4}>
                                                        <Text size="xs" c="dimmed">
                                                            ⏱️ {chunk.processing_time}s
                                                        </Text>
                                                        <Text size="xs" c="dimmed">
                                                            📊 {chunk.entries_count} entries
                                                        </Text>
                                                        {(chunk as any).duplicates_removed > 0 && (
                                                            <Text size="xs" c="orange">
                                                                🔄 {(chunk as any).duplicates_removed} duplicates removed
                                                            </Text>
                                                        )}
                                                    </Group>
                                                    {chunk.error && (
                                                        <Alert color="red" variant="light" mt="xs">
                                                            <Text size="xs">{chunk.error}</Text>
                                                        </Alert>
                                                    )}
                                                </Timeline.Item>
                                            ))}
                                        </Timeline>
                                    </Stack>
                                </Card>
                            )}

                            {/* Auto-Feedback Quick Actions */}
                            {result && processingStatus === 'complete' && result.session_id && !feedbackSubmitted && !cardDismissed && autoFeedback && !autoFeedback.failed && (
                                <Card shadow="sm" padding="lg" radius="lg" withBorder style={{ background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
                                    {!autoFeedback || autoFeedback.rating === 0 ? (
                                        <Group justify="center" p="md">
                                            <Loader size="sm" color="violet" />
                                            <Text size="sm" c="dimmed">Loading auto-feedback...</Text>
                                        </Group>
                                    ) : (
                                        <Stack gap="md">
                                            <Group>
                                                <ThemeIcon size="lg" radius="md" variant="light" color="violet">
                                                    <IconStar size={20} />
                                                </ThemeIcon>
                                                <div style={{ flex: 1 }}>
                                                    <Text size="md" fw={600}>
                                                        🤖 System Rating: {autoFeedback.rating}⭐
                                                    </Text>
                                                    <Text size="xs" c="dimmed" mt={4}>
                                                        {autoFeedback.feedback_text}
                                                    </Text>
                                                </div>
                                                <ActionIcon
                                                    variant="subtle"
                                                    color="gray"
                                                    onClick={() => setCardDismissed(true)}
                                                    size="sm"
                                                >
                                                    ✕
                                                </ActionIcon>
                                            </Group>

                                            <Divider />

                                            <div>
                                                <Text size="sm" fw={500} mb="sm">
                                                    Do you agree with this rating?
                                                </Text>
                                                <Group gap="sm">
                                                    <Button
                                                        leftSection={<IconCheck size={18} />}
                                                        color="green"
                                                        variant="light"
                                                        onClick={handleAgreeWithRating}
                                                        loading={submittingFeedback}
                                                        disabled={!autoFeedback || autoFeedback.rating === 0}
                                                    >
                                                        Agree ✓
                                                    </Button>
                                                    <Button
                                                        leftSection={<IconMessageCircle size={18} />}
                                                        color="orange"
                                                        variant="light"
                                                        onClick={handleOpenFeedback}
                                                    >
                                                        Disagree - Provide Feedback
                                                    </Button>
                                                </Group>
                                            </div>
                                        </Stack>
                                    )}
                                </Card>
                            )}

                            {/* Results */}
                            {result && processingStatus === 'complete' && (
                                <Card shadow="sm" padding="lg" radius="lg" withBorder>
                                    <Stack gap="md">
                                        <Group justify="space-between">
                                            <Group>
                                                <ThemeIcon size="lg" radius="md" variant="light" color="teal">
                                                    <IconFileAnalytics size={20} />
                                                </ThemeIcon>
                                                <Text size="lg" fw={600}>
                                                    Extraction Results
                                                </Text>
                                            </Group>
                                            <DownloadButtons
                                                onDownloadExcel={handleDownloadExcel}
                                                onDownloadJSON={handleDownloadJSON}
                                                disabled={isDownloading}
                                            />
                                        </Group>

                                        <Divider />

                                        <Tabs defaultValue="data" color="violet">
                                            <Tabs.List>
                                                <Tabs.Tab value="data">
                                                    📊 Structured Data
                                                </Tabs.Tab>
                                                <Tabs.Tab value="json">
                                                    🔍 Raw JSON
                                                </Tabs.Tab>
                                                {selectedFile && (
                                                    <Tabs.Tab value="pdf">
                                                        📄 Original PDF
                                                    </Tabs.Tab>
                                                )}
                                                {chunkInfo && (
                                                    <Tabs.Tab value="chunks">
                                                        📦 Chunks
                                                    </Tabs.Tab>
                                                )}
                                            </Tabs.List>

                                            <Tabs.Panel value="data" pt="md">
                                                <DataTable data={result.entries} />
                                            </Tabs.Panel>

                                            <Tabs.Panel value="json" pt="md">
                                                <JSONViewer data={rawJSON} />
                                            </Tabs.Panel>

                                            {selectedFile && (
                                                <Tabs.Panel value="pdf" pt="md">
                                                    <Card withBorder style={{ height: '600px', overflow: 'hidden' }}>
                                                        <Stack gap="xs" h="100%">
                                                            <Group justify="space-between" p="sm">
                                                                <Text size="sm" fw={500} c="dimmed">
                                                                    📄 {selectedFile.name}
                                                                </Text>
                                                                <Badge size="sm" color="blue">
                                                                    {(selectedFile.size / 1024).toFixed(1)} KB
                                                                </Badge>
                                                            </Group>
                                                            <Divider />
                                                            <div style={{ flex: 1, position: 'relative' }}>
                                                                <iframe
                                                                    src={URL.createObjectURL(selectedFile)}
                                                                    style={{
                                                                        width: '100%',
                                                                        height: '100%',
                                                                        border: 'none',
                                                                        borderRadius: '8px'
                                                                    }}
                                                                    title="PDF Preview"
                                                                />
                                                            </div>
                                                        </Stack>
                                                    </Card>
                                                </Tabs.Panel>
                                            )}

                                            {chunkInfo && (
                                                <Tabs.Panel value="chunks" pt="md">
                                                    <Stack gap="md">
                                                        {chunkInfo.chunk_results.map((chunkResult) => (
                                                            <Card key={chunkResult.chunk_id} padding="md" withBorder>
                                                                <Stack gap="xs">
                                                                    <Group justify="space-between">
                                                                        <Text size="sm" fw={600} c="violet">
                                                                            Chunk {chunkResult.chunk_id}
                                                                        </Text>
                                                                        <Badge size="sm" color="blue">
                                                                            {chunkResult.entries.length} entries
                                                                        </Badge>
                                                                    </Group>
                                                                    <DataTable data={chunkResult.entries} />
                                                                </Stack>
                                                            </Card>
                                                        ))}
                                                    </Stack>
                                                </Tabs.Panel>
                                            )}
                                        </Tabs>
                                    </Stack>
                                </Card>
                            )}
                        </Stack>
                    </Grid.Col>

                    {/* Right Column - Stats & Info */}
                    <Grid.Col span={{ base: 12, md: 4 }}>
                        <Stack gap="lg">
                            <Card shadow="sm" padding="lg" radius="lg" withBorder>
                                <Stack gap="md">
                                    <Text size="lg" fw={600}>
                                        📊 Quick Stats
                                    </Text>
                                    <Group justify="space-between">
                                        <Group gap="xs">
                                            <Text size="sm" c="dimmed">Backend Status</Text>
                                            {backendStatus === 'checking' && <IconLoader size={14} />}
                                            {backendStatus === 'online' && <IconWifi size={14} color="green" />}
                                            {(backendStatus === 'offline' || backendStatus === 'error') && <IconWifiOff size={14} color="red" />}
                                        </Group>
                                        <Group gap="xs">
                                            <Text 
                                                size="sm" 
                                                fw={500} 
                                                c={
                                                    backendStatus === 'online' ? 'green' : 
                                                    backendStatus === 'checking' ? 'yellow' : 
                                                    'red'
                                                }
                                            >
                                                {backendStatus === 'checking' ? 'Checking...' :
                                                 backendStatus === 'online' ? 'Online' :
                                                 backendStatus === 'offline' ? 'Offline' : 'Error'}
                                            </Text>
                                            {(backendStatus === 'offline' || backendStatus === 'error') && (
                                                <ActionIcon 
                                                    size="sm" 
                                                    variant="subtle" 
                                                    onClick={refreshBackend}
                                                    title="Retry connection"
                                                >
                                                    <IconRefresh size={12} />
                                                </ActionIcon>
                                            )}
                                        </Group>
                                    </Group>
                                    {selectedFile && (
                                        <Group justify="space-between">
                                            <Text size="sm" c="dimmed">File Status</Text>
                                            <Text size="sm" fw={500} c="violet">
                                                Ready
                                            </Text>
                                        </Group>
                                    )}
                                    {result && (
                                        <>
                                            <Group justify="space-between">
                                                <Text size="sm" c="dimmed">Rows Extracted</Text>
                                                <Text size="sm" fw={500}>{result.entries.length}</Text>
                                            </Group>
                                            {result.processing_time && (
                                                <Group justify="space-between">
                                                    <Text size="sm" c="dimmed">Processing Time</Text>
                                                    <Badge color="blue" variant="light" leftSection={<IconClock size={12} />}>
                                                        {result.processing_time}s
                                                    </Badge>
                                                </Group>
                                            )}
                                            {result.global_notes && (
                                                <Alert color="blue" variant="light" mt="xs" title="Global Note">
                                                    <Text size="xs">{result.global_notes}</Text>
                                                </Alert>
                                            )}
                                        </>
                                    )}
                                </Stack>
                            </Card>

                            <Card shadow="sm" padding="lg" radius="lg" withBorder style={{ background: 'linear-gradient(135deg, #f6f0ff 0%, #ffffff 100%)' }}>
                                <Stack gap="sm">
                                    <Text size="md" fw={600} c="violet">
                                        ✨ Key Features
                                    </Text>
                                    <Group gap="xs">
                                        <ThemeIcon size="xs" color="violet" radius="xl"><IconCheck size={10} /></ThemeIcon>
                                        <Text size="sm">Dynamic column detection</Text>
                                    </Group>
                                    <Group gap="xs">
                                        <ThemeIcon size="xs" color="violet" radius="xl"><IconCheck size={10} /></ThemeIcon>
                                        <Text size="sm">100% data fidelity</Text>
                                    </Group>
                                    <Group gap="xs">
                                        <ThemeIcon size="xs" color="violet" radius="xl"><IconCheck size={10} /></ThemeIcon>
                                        <Text size="sm">Context-aware extraction</Text>
                                    </Group>
                                </Stack>
                            </Card>
                        </Stack>
                    </Grid.Col>
                </Grid>
            </Stack>

            {/* Analytics Modal */}
            <Modal
                opened={showAnalytics}
                onClose={() => setShowAnalytics(false)}
                title={
                    <Group>
                        <IconChartBar size={24} />
                        <Text size="lg" fw={600}>System Analytics</Text>
                    </Group>
                }
                size="lg"
            >
                {analytics && analytics.enabled ? (
                    <Stack gap="md">
                        <Paper p="md" withBorder>
                            <Text size="sm" fw={600} mb="xs" c="dimmed">SYSTEM STATISTICS</Text>
                            <Grid>
                                <Grid.Col span={6}>
                                    <Stack gap="xs">
                                        <Text size="sm" c="dimmed">Total Sessions</Text>
                                        <Text size="xl" fw={700}>{analytics.analytics?.total_sessions || 0}</Text>
                                    </Stack>
                                </Grid.Col>
                                <Grid.Col span={6}>
                                    <Stack gap="xs">
                                        <Text size="sm" c="dimmed">Success Rate</Text>
                                        <Text size="xl" fw={700} c="green">{analytics.analytics?.success_rate || 0}%</Text>
                                    </Stack>
                                </Grid.Col>
                                <Grid.Col span={6}>
                                    <Stack gap="xs">
                                        <Text size="sm" c="dimmed">Total Feedback</Text>
                                        <Text size="xl" fw={700}>{analytics.analytics?.total_feedback || 0}</Text>
                                    </Stack>
                                </Grid.Col>
                                <Grid.Col span={6}>
                                    <Stack gap="xs">
                                        <Text size="sm" c="dimmed">Combined Avg</Text>
                                        <Group gap="xs">
                                            <Text size="xl" fw={700}>{analytics.analytics?.avg_user_rating || 0}</Text>
                                            <IconStar size={20} fill="gold" color="gold" />
                                        </Group>
                                    </Stack>
                                </Grid.Col>
                            </Grid>
                        </Paper>

                        {analytics.analytics?.feedback_breakdown && (
                            <Paper p="md" withBorder style={{ background: 'rgba(124, 58, 237, 0.05)' }}>
                                <Text size="sm" fw={600} mb="md" c="dimmed">RATING BREAKDOWN</Text>
                                <Grid>
                                    <Grid.Col span={6}>
                                        <Paper p="sm" withBorder style={{ background: 'white' }}>
                                            <Stack gap="xs">
                                                <Group gap="xs">
                                                    <Text size="xs" c="dimmed">🤖 Automated</Text>
                                                    <Badge size="xs" color="violet" variant="light">
                                                        {analytics.analytics.feedback_breakdown.auto_feedback_count}
                                                    </Badge>
                                                </Group>
                                                <Group gap="xs">
                                                    <Text size="lg" fw={700}>{analytics.analytics.feedback_breakdown.avg_auto_rating}</Text>
                                                    <IconStar size={16} fill="gold" color="gold" />
                                                </Group>
                                                <Text size="xs" c="dimmed">
                                                    {((analytics.analytics.feedback_breakdown.auto_feedback_count / analytics.analytics.total_feedback) * 100).toFixed(1)}% of feedback
                                                </Text>
                                            </Stack>
                                        </Paper>
                                    </Grid.Col>
                                    <Grid.Col span={6}>
                                        <Paper p="sm" withBorder style={{ background: 'white' }}>
                                            <Stack gap="xs">
                                                <Group gap="xs">
                                                    <Text size="xs" c="dimmed">👤 User Override</Text>
                                                    <Badge size="xs" color="orange" variant="light">
                                                        {analytics.analytics.feedback_breakdown.user_feedback_count}
                                                    </Badge>
                                                </Group>
                                                <Group gap="xs">
                                                    <Text size="lg" fw={700}>{analytics.analytics.feedback_breakdown.avg_user_rating || 'N/A'}</Text>
                                                    {analytics.analytics.feedback_breakdown.avg_user_rating > 0 && (
                                                        <IconStar size={16} fill="gold" color="gold" />
                                                    )}
                                                </Group>
                                                <Text size="xs" c="dimmed">
                                                    {analytics.analytics.feedback_breakdown.override_rate}% override rate
                                                </Text>
                                            </Stack>
                                        </Paper>
                                    </Grid.Col>
                                </Grid>
                                <Divider my="sm" />
                                <Group justify="space-between">
                                    <Text size="xs" c="dimmed">Agreement Rate</Text>
                                    <Badge color="green" variant="light">
                                        {analytics.analytics.feedback_breakdown.agreement_rate}%
                                    </Badge>
                                </Group>
                            </Paper>
                        )}

                        {analytics.trends && Object.keys(analytics.trends).length > 0 ? (
                            <Paper p="md" withBorder>
                                <Text size="sm" fw={600} mb="xs" c="dimmed">PERFORMANCE TRENDS (LAST 7 DAYS)</Text>
                                <Stack gap="sm">
                                    <Group justify="space-between">
                                        <Text size="sm">Avg Processing Time</Text>
                                        <Badge color="blue">{analytics.trends.avg_processing_time?.toFixed(2) || '0.00'}s</Badge>
                                    </Group>
                                    <Group justify="space-between">
                                        <Text size="sm">Avg Entries Extracted</Text>
                                        <Badge color="violet">{Math.round(analytics.trends.avg_entries || 0)}</Badge>
                                    </Group>
                                    <Group justify="space-between">
                                        <Text size="sm">Avg Quality Score</Text>
                                        <Badge color="green">{analytics.trends.avg_quality?.toFixed(2) || '0.00'}</Badge>
                                    </Group>
                                    <Group justify="space-between">
                                        <Text size="sm">Total Sessions</Text>
                                        <Badge color="gray">{analytics.trends.total_sessions || 0}</Badge>
                                    </Group>
                                </Stack>
                            </Paper>
                        ) : (
                            <Paper p="md" withBorder>
                                <Text size="sm" fw={600} mb="xs" c="dimmed">PERFORMANCE TRENDS (LAST 7 DAYS)</Text>
                                <Alert color="blue" variant="light">
                                    No trend data available yet. Process more documents to see performance trends.
                                </Alert>
                            </Paper>
                        )}

                        <Paper p="md" withBorder>
                            <Text size="sm" fw={600} mb="xs" c="dimmed">LEARNING SYSTEM</Text>
                            <Stack gap="xs">
                                <Group justify="space-between">
                                    <Text size="sm">Learned Patterns</Text>
                                    <Badge color="indigo">{analytics.analytics?.total_patterns || 0}</Badge>
                                </Group>
                                <Group justify="space-between">
                                    <Text size="sm">Active Optimizations</Text>
                                    <Badge color="teal">{analytics.analytics?.active_optimizations || 0}</Badge>
                                </Group>
                            </Stack>
                        </Paper>
                    </Stack>
                ) : (
                    <Alert color="orange" title="Learning System Disabled">
                        The learning system is not enabled. Enable it in the backend configuration to track analytics.
                    </Alert>
                )}
            </Modal>

            {/* Feedback Modal */}
            <Modal
                opened={showFeedback}
                onClose={() => {
                    setShowFeedback(false);
                    setAutoFeedback(null);
                    setFeedbackRating(0);
                    setFeedbackText('');
                }}
                title={
                    <Group>
                        <IconMessageCircle size={24} />
                        <Text size="lg" fw={600}>
                            {autoFeedback ? 'Override Auto-Feedback' : 'Submit Feedback'}
                        </Text>
                    </Group>
                }
                size="md"
            >
                <Stack gap="md">
                    {loadingAutoFeedback ? (
                        <Group justify="center" p="xl">
                            <Loader size="sm" />
                            <Text size="sm" c="dimmed">Loading auto-feedback...</Text>
                        </Group>
                    ) : (
                        <>
                            {result?.session_id ? (
                                <Alert color="blue" variant="light">
                                    Session ID: <Text span fw={600}>{result.session_id}</Text>
                                </Alert>
                            ) : (
                                <Alert color="orange" variant="light">
                                    No active session. Feedback will be submitted as general feedback.
                                </Alert>
                            )}

                            {autoFeedback && (
                                <Paper p="md" withBorder style={{ background: 'rgba(124, 58, 237, 0.05)' }}>
                                    <Stack gap="xs">
                                        <Group justify="space-between">
                                            <Text size="sm" fw={600} c="violet">
                                                🤖 Automated Rating
                                            </Text>
                                            <Group gap="xs">
                                                {[...Array(autoFeedback.rating)].map((_, i) => (
                                                    <IconStar key={i} size={16} fill="gold" color="gold" />
                                                ))}
                                                <Text size="sm" fw={600}>{autoFeedback.rating}/5</Text>
                                            </Group>
                                        </Group>
                                        <Text size="xs" c="dimmed">
                                            {autoFeedback.feedback_text}
                                        </Text>
                                        <Divider />
                                        <Text size="xs" c="dimmed" fs="italic">
                                            This session was automatically rated based on quality metrics. You can override this rating below if you disagree.
                                        </Text>
                                    </Stack>
                                </Paper>
                            )}

                            <div>
                                <Text size="sm" fw={500} mb="xs">
                                    {autoFeedback ? 'Your Rating (Override)' : 'Rate your experience'}
                                </Text>
                                <Group gap="xs">
                                    {[1, 2, 3, 4, 5].map((star) => (
                                        <ActionIcon
                                            key={star}
                                            size="xl"
                                            variant={feedbackRating >= star ? 'filled' : 'light'}
                                            color="yellow"
                                            onClick={() => setFeedbackRating(star)}
                                        >
                                            <IconStar size={24} fill={feedbackRating >= star ? 'gold' : 'none'} />
                                        </ActionIcon>
                                    ))}
                                </Group>
                                {autoFeedback && feedbackRating !== autoFeedback.rating && feedbackRating > 0 && (
                                    <Alert color="orange" variant="light" mt="xs">
                                        <Text size="xs">
                                            You're changing the rating from {autoFeedback.rating}⭐ to {feedbackRating}⭐
                                        </Text>
                                    </Alert>
                                )}
                            </div>

                            <Textarea
                                label="Additional Comments (Optional)"
                                placeholder={autoFeedback ? "Explain why you're changing the rating..." : "Tell us about your experience..."}
                                value={feedbackText}
                                onChange={(e) => setFeedbackText(e.currentTarget.value)}
                                minRows={4}
                            />

                            <Group justify="flex-end" gap="sm">
                                <Button variant="light" onClick={() => {
                                    setShowFeedback(false);
                                    setAutoFeedback(null);
                                    setFeedbackRating(0);
                                    setFeedbackText('');
                                }}>
                                    Cancel
                                </Button>
                                <Button
                                    onClick={handleSubmitFeedback}
                                    loading={submittingFeedback}
                                    disabled={feedbackRating === 0}
                                    color={autoFeedback && feedbackRating !== autoFeedback.rating ? 'orange' : 'violet'}
                                >
                                    {autoFeedback && feedbackRating !== autoFeedback.rating ? 'Submit Override' : 'Submit Feedback'}
                                </Button>
                            </Group>
                        </>
                    )}
                </Stack>
            </Modal>
        </Container>
    );
}
