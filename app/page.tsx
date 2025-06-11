import LanguageToolDemo from './components/language-tool-demo';

const Home = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <main className="container mx-auto py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Writeo</h1>
          <p className="text-gray-600">
            AI-powered writing assistant with LanguageTool integration
          </p>
        </div>
        <LanguageToolDemo />
      </main>
    </div>
  );
};

export default Home;
