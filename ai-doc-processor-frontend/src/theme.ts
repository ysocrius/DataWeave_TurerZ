import { createTheme, rem } from '@mantine/core';

export const theme = createTheme({
    primaryColor: 'violet',
    colors: {
        violet: [
            '#f6f0ff',
            '#e9dbff',
            '#d1b3ff',
            '#b888ff',
            '#a364ff',
            '#954dff',
            '#8e41ff',
            '#7b31e6',
            '#6d2bce',
            '#5f23b6',
        ],
        dark: [
            '#C1C2C5',
            '#A6A7AB',
            '#909296',
            '#5c5f66',
            '#373A40',
            '#2C2E33',
            '#25262b',
            '#1A1B1E',
            '#141517',
            '#101113',
        ],
    },
    shadows: {
        md: '1px 1px 3px rgba(0, 0, 0, .25)',
        xl: '5px 5px 3px rgba(0, 0, 0, .25)',
    },
    fontFamily: 'Inter, sans-serif',
    headings: {
        fontFamily: 'Inter, sans-serif',
        sizes: {
            h1: { fontSize: rem(36), lineHeight: rem(1.4) },
        },
    },
    components: {
        Card: {
            defaultProps: {
                radius: 'md',
                withBorder: true,
            },
            styles: (theme: any) => ({

                root: {
                    backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : '#ffffff',
                    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                    '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: theme.shadows.md,
                    },
                },
            }),
        },
        Button: {
            defaultProps: {
                radius: 'md',
            },
        },
    },
});
