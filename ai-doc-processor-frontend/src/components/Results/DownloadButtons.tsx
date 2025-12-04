import { Button, Group } from '@mantine/core';
import { IconDownload, IconFileSpreadsheet, IconJson } from '@tabler/icons-react';

interface DownloadButtonsProps {
    onDownloadExcel: () => void;
    onDownloadJSON: () => void;
    disabled?: boolean;
}

export function DownloadButtons({
    onDownloadExcel,
    onDownloadJSON,
    disabled = false
}: DownloadButtonsProps) {
    return (
        <Group>
            <Button
                leftSection={<IconFileSpreadsheet size={18} />}
                onClick={onDownloadExcel}
                disabled={disabled}
                size="md"
            >
                Download Excel
            </Button>
            <Button
                leftSection={<IconJson size={18} />}
                onClick={onDownloadJSON}
                disabled={disabled}
                variant="light"
                size="md"
            >
                Download JSON
            </Button>
        </Group>
    );
}
