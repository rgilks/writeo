import LanguageToolDemo from './components/language-tool-demo';

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-r from-gray-900 via-slate-800 to-gray-900 text-white animate-gradient-bg relative overflow-hidden">
      <main className="container mx-auto py-10 sm:py-16 relative z-10">
        <div className="text-center mb-10 sm:mb-16">
          <h1 className="font-display text-5xl sm:text-6xl md:text-7xl font-extrabold mb-4 bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 bg-clip-text text-transparent">
            writeo
          </h1>

          <p className="max-w-2xl mx-auto text-lg sm:text-xl text-gray-400 font-sans">
            Your AI-powered writing assistant.
          </p>
        </div>
        <LanguageToolDemo />
      </main>
    </div>
  );
};

export default Home;
