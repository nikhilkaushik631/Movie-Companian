import './globals.css';

export const metadata = {
  title: 'Cinemizer - Your Movie Companion',
  description: 'AI-powered discovery for movies and TV shows',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="app-body" style={{ margin: 0 }}>
        {children}
      </body>
    </html>
  );
}


