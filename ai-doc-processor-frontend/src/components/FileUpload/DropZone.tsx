import { useState, useCallback } from 'react';
import { Group, Text, rem, Stack } from '@mantine/core';
import { IconUpload, IconFile, IconX } from '@tabler/icons-react';
import { Dropzone, PDF_MIME_TYPE } from '@mantine/dropzone';


interface FileDropZoneProps {
  onFileSelect: (file: File) => void;
  maxSize?: number;
  disabled?: boolean;
}

export function FileDropZone({
  onFileSelect,
  maxSize = 10 * 1024 * 1024, // 10MB default
  disabled = false
}: FileDropZoneProps) {
  const [error, setError] = useState<string | null>(null);

  const handleDrop = useCallback((files: File[]) => {
    setError(null);
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const handleReject = useCallback((rejections: any[]) => {
    if (rejections.length > 0) {
      const rejection = rejections[0];
      if (rejection.errors[0]?.code === 'file-too-large') {
        setError(`File is too large. Maximum size is ${maxSize / 1024 / 1024}MB`);
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        setError('Invalid file type. Please upload a PDF file.');
      } else {
        setError('Failed to upload file. Please try again.');
      }
    }
  }, [maxSize]);

  return (
    <Stack gap="md">
      <Dropzone
        onDrop={handleDrop}
        onReject={handleReject}
        maxSize={maxSize}
        accept={PDF_MIME_TYPE}
        disabled={disabled}
        multiple={false}
      >
        <Group justify="center" gap="xl" mih={220} style={{ pointerEvents: 'none' }}>
          <Dropzone.Accept>
            <IconUpload
              style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-blue-6)' }}
              stroke={1.5}
            />
          </Dropzone.Accept>
          <Dropzone.Reject>
            <IconX
              style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-red-6)' }}
              stroke={1.5}
            />
          </Dropzone.Reject>
          <Dropzone.Idle>
            <IconFile
              style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-dimmed)' }}
              stroke={1.5}
            />
          </Dropzone.Idle>

          <div>
            <Text size="xl" inline>
              Drag PDF here or click to select
            </Text>
            <Text size="sm" c="dimmed" inline mt={7}>
              Upload a PDF document to extract structured data
            </Text>
            <Text size="xs" c="dimmed" mt="xs">
              Maximum file size: {maxSize / 1024 / 1024}MB
            </Text>
          </div>
        </Group>
      </Dropzone>

      {error && (
        <Text c="red" size="sm">
          {error}
        </Text>
      )}
    </Stack>
  );
}
