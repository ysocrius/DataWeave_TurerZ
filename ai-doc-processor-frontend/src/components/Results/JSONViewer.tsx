import { Code, ScrollArea } from '@mantine/core';

interface JSONViewerProps {
    data: any;
    maxHeight?: number;
}

export function JSONViewer({ data, maxHeight = 400 }: JSONViewerProps) {
    const jsonString = typeof data === 'string'
        ? data
        : JSON.stringify(data, null, 2);

    return (
        <ScrollArea h={maxHeight}>
            <Code block>
                {jsonString}
            </Code>
        </ScrollArea>
    );
}
