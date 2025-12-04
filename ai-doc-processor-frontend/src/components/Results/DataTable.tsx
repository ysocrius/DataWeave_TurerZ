import { Table, ScrollArea, Text, Badge } from '@mantine/core';

interface DataRow {
    '#'?: number;
    Key: string;
    Value: string;
    Comments?: string;
}

interface DataTableProps {
    data: DataRow[];
    maxHeight?: number;
}

export function DataTable({ data, maxHeight = 500 }: DataTableProps) {
    if (!data || data.length === 0) {
        return (
            <Text c="dimmed" ta="center" py="xl">
                No data to display
            </Text>
        );
    }

    const hasComments = data.some(row => row.Comments);
    const hasNumbers = data.some(row => row['#'] !== undefined);

    return (
        <ScrollArea h={maxHeight}>
            <Table striped highlightOnHover withTableBorder withColumnBorders>
                <Table.Thead>
                    <Table.Tr>
                        {hasNumbers && <Table.Th>#</Table.Th>}
                        <Table.Th>Key</Table.Th>
                        <Table.Th>Value</Table.Th>
                        {hasComments && <Table.Th>Comments</Table.Th>}
                    </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                    {data.map((row, index) => (
                        <Table.Tr key={index}>
                            {hasNumbers && (
                                <Table.Td>
                                    <Text size="sm" c="dimmed">
                                        {row['#'] || index + 1}
                                    </Text>
                                </Table.Td>
                            )}
                            <Table.Td>
                                <Text size="sm" fw={500}>
                                    {row.Key}
                                </Text>
                            </Table.Td>
                            <Table.Td>
                                <Text size="sm">
                                    {row.Value}
                                </Text>
                            </Table.Td>
                            {hasComments && (
                                <Table.Td>
                                    {row.Comments && (
                                        <Text size="xs" c="dimmed">
                                            {row.Comments.includes('[GLOBAL NOTE:') ? (
                                                <Badge color="blue" variant="light" size="sm">
                                                    {row.Comments}
                                                </Badge>
                                            ) : (
                                                row.Comments
                                            )}
                                        </Text>
                                    )}
                                </Table.Td>
                            )}
                        </Table.Tr>
                    ))}
                </Table.Tbody>
            </Table>
        </ScrollArea>
    );
}
