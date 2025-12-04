import { Card, Text, Stack, Group, ThemeIcon, Loader } from '@mantine/core';
import { IconCheck, IconAlertCircle, IconClock } from '@tabler/icons-react';
import { useState, useEffect } from 'react';

export type ProcessingStatus = 'idle' | 'extracting' | 'analyzing' | 'structuring' | 'complete' | 'error';

interface ProcessingStep {
    label: string;
    status: 'pending' | 'active' | 'complete' | 'error';
}

interface StatusCardProps {
    status: ProcessingStatus;
    error?: string;
    steps?: ProcessingStep[];
}

function StepTimer({ active }: { active: boolean }) {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (active) {
            interval = setInterval(() => {
                setSeconds(s => s + 1);
            }, 1000);
        } else {
            setSeconds(0);
        }
        return () => clearInterval(interval);
    }, [active]);

    if (!active) return null;

    return (
        <Text size="xs" c="violet" fw={500} style={{ fontVariantNumeric: 'tabular-nums' }}>
            ({seconds}s)
        </Text>
    );
}

export function StatusCard({ status, error, steps }: StatusCardProps) {
    const defaultSteps: ProcessingStep[] = [
        {
            label: 'Extracting text from PDF',
            status: status === 'extracting' ? 'active' : status === 'idle' ? 'pending' : 'complete'
        },
        {
            label: 'Analyzing with AI',
            status: status === 'analyzing' ? 'active' : ['idle', 'extracting'].includes(status) ? 'pending' : 'complete'
        },
        {
            label: 'Structuring data',
            status: status === 'structuring' ? 'active' : status === 'complete' ? 'complete' : 'pending'
        },
    ];

    const displaySteps = steps || defaultSteps;

    const getStepIcon = (stepStatus: string) => {
        switch (stepStatus) {
            case 'complete':
                return <ThemeIcon color="teal" size={24} radius="xl" variant="light"><IconCheck size={16} /></ThemeIcon>;
            case 'active':
                return <ThemeIcon color="violet" size={24} radius="xl" variant="light"><Loader size={14} color="violet" /></ThemeIcon>;
            case 'error':
                return <ThemeIcon color="red" size={24} radius="xl" variant="light"><IconAlertCircle size={16} /></ThemeIcon>;
            default:
                return <ThemeIcon color="gray" size={24} radius="xl" variant="subtle"><IconClock size={16} /></ThemeIcon>;
        }
    };

    return (
        <Card shadow="sm" padding="lg" radius="lg" withBorder>
            <Stack gap="md">
                <Text size="lg" fw={600}>
                    {status === 'complete' ? '✅ Processing Complete' :
                        status === 'error' ? '❌ Processing Failed' :
                            '🤖 AI Processing'}
                </Text>

                <Stack gap="sm">
                    {displaySteps.map((step, index) => (
                        <Group key={index} gap="md">
                            {getStepIcon(step.status)}
                            <Group gap="xs">
                                <Text
                                    size="sm"
                                    c={step.status === 'complete' ? 'dimmed' : step.status === 'active' ? 'violet' : 'dimmed'}
                                    fw={step.status === 'active' ? 600 : 400}
                                >
                                    {step.label}
                                </Text>
                                <StepTimer active={step.status === 'active'} />
                            </Group>
                        </Group>
                    ))}
                </Stack>

                {error && (
                    <Text c="red" size="sm" bg="red.0" p="xs" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                        Error: {error}
                    </Text>
                )}
            </Stack>
        </Card>
    );
}
