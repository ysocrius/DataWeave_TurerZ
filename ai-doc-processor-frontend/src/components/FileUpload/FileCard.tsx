import { Card, Group, Text, Badge, ActionIcon, Stack, Progress } from '@mantine/core';
import { IconFile, IconTrash } from '@tabler/icons-react';

interface FileInfo {
    name: string;
    size: number;
    uploadProgress?: number;
}

interface FileCardProps {
    file: FileInfo;
    onRemove?: () => void;
    showProgress?: boolean;
}

export function FileCard({ file, onRemove, showProgress = false }: FileCardProps) {
    const formatFileSize = (bytes: number): string => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    };

    return (
        <Card shadow="sm" padding="md" radius="md" withBorder>
            <Stack gap="xs">
                <Group justify="space-between">
                    <Group gap="sm">
                        <IconFile size={24} />
                        <div>
                            <Text size="sm" fw={500}>
                                {file.name}
                            </Text>
                            <Text size="xs" c="dimmed">
                                {formatFileSize(file.size)}
                            </Text>
                        </div>
                    </Group>
                    <Group gap="xs">
                        <Badge color="blue" variant="light">
                            PDF
                        </Badge>
                        {onRemove && (
                            <ActionIcon color="red" variant="subtle" onClick={onRemove}>
                                <IconTrash size={18} />
                            </ActionIcon>
                        )}
                    </Group>
                </Group>

                {showProgress && file.uploadProgress !== undefined && (
                    <Progress value={file.uploadProgress} size="sm" />
                )}
            </Stack>
        </Card>
    );
}
