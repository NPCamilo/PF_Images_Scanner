import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Nexus-DSP | Scanner de Imagen, Audio y Bit',
  description:
    'Estación de trabajo DSP Client-Side para Comunicaciones I — Universidad Tecnológica de Pereira',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className="dark">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased selection:bg-cyan-500/30">
        {children}
      </body>
    </html>
  );
}
