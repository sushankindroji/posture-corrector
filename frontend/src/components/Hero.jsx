import { Link } from 'react-scroll';

function Hero() {
  return (
    <section id="home" className="min-h-screen flex items-center pt-16 relative bg-hero-pattern">
      <div className="absolute inset-0 bg-gradient-to-b from-background/80 to-background z-0"></div>
      
      {/* Hero content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 py-24">
        <div className="text-center md:text-left md:flex md:items-center md:justify-between">
          <div className="md:w-1/2 mb-10 md:mb-0">
            <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl md:text-6xl mb-4">
              <span className="block">Perfect Your</span>
              <span className="gradient-text">Posture</span>
            </h1>
            <p className="mt-4 max-w-lg mx-auto md:mx-0 text-lg text-text-secondary">
              Maintain correct posture in everyday activities, workouts, and yoga practice.
              Our comprehensive guide helps you avoid pain and improve your physical well-being.
            </p>
            <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
              <Link
                to="postures"
                spy={true}
                smooth={true}
                offset={-70}
                duration={500}
                className="btn-primary"
              >
                <span>Check Your Posture</span>
                <svg className="ml-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </Link>
              <Link
                to="features"
                spy={true}
                smooth={true}
                offset={-70}
                duration={500}
                className="btn-outline"
              >
                Learn More
              </Link>
            </div>
          </div>
          
          <div className="md:w-1/2 flex justify-center">
            <div className="relative w-full max-w-md">
              {/* Abstract SVG graphic */}
              <svg className="w-full h-auto" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                <path d="M296.3,200c0,53.2-43.1,96.3-96.3,96.3S103.7,253.2,103.7,200s43.1-96.3,96.3-96.3S296.3,146.8,296.3,200z" 
                      fill="none" stroke="#6366f1" strokeWidth="4" strokeDasharray="10,6" />
                <path d="M323.7,200c0,68.3-55.4,123.7-123.7,123.7S76.3,268.3,76.3,200S131.7,76.3,200,76.3S323.7,131.7,323.7,200z" 
                      fill="none" stroke="#4f46e5" strokeWidth="3" strokeDasharray="8,8" opacity="0.7" />
                <path d="M200,200 L200,120" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
                <path d="M200,200 L260,200" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
                <circle cx="200" cy="200" r="10" fill="#10b981" />
                <circle cx="200" cy="120" r="6" fill="#6366f1" />
                <circle cx="260" cy="200" r="6" fill="#6366f1" />
                <path d="M150,270 L250,270 L200,320 Z" fill="#4f46e5" opacity="0.6" />
                <line x1="100" y1="150" x2="120" y2="170" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
                <line x1="100" y1="170" x2="120" y2="150" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
                <line x1="280" y1="150" x2="300" y2="170" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
                <line x1="280" y1="170" x2="300" y2="150" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
              </svg>
            </div>
          </div>
        </div>
      </div>
      
      {/* Wave separator */}
      <div className="absolute bottom-0 left-0 right-0">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" className="w-full h-auto">
          <path fill="#1e293b" fillOpacity="1" d="M0,128L48,149.3C96,171,192,213,288,224C384,235,480,213,576,186.7C672,160,768,128,864,122.7C960,117,1056,139,1152,138.7C1248,139,1344,117,1392,106.7L1440,96L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
        </svg>
      </div>
    </section>
  );
}

export default Hero;