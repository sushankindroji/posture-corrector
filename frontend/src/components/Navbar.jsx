import { useState, useEffect } from 'react';
import { Link } from 'react-scroll';
import logo from '../assets/logo.svg';

function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 20) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${scrolled ? 'bg-background/90 backdrop-blur-md shadow-lg' : 'bg-transparent'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <img className="h-8 w-8 text-primary" src={logo} alt="Logo" />
              <span className="ml-2 text-xl font-bold text-text-primary">Posture<span className="text-primary">Corrector</span></span>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6">
            <Link
              to="home"
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
              className="px-3 py-2 text-text-primary hover:text-primary transition-colors cursor-pointer font-medium"
            >
              Home
            </Link>
            <Link
              to="features"
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
              className="px-3 py-2 text-text-primary hover:text-primary transition-colors cursor-pointer font-medium"
            >
              Features
            </Link>
            <Link
              to="postures"
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
              className="px-5 py-2 rounded-md bg-primary text-text-primary hover:bg-secondary transition-all duration-300 cursor-pointer font-medium shadow-lg hover:shadow-xl"
            >
              Posture Check
            </Link>
          </div>
          
          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-text-secondary hover:text-primary hover:bg-background-light focus:outline-none transition-colors"
            >
              <svg
                className="h-6 w-6"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {isMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-background-light/90 backdrop-blur-md">
            <Link
              to="home"
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
              className="block px-3 py-2 text-text-primary hover:text-primary hover:bg-background-card rounded-md transition-colors cursor-pointer"
              onClick={() => setIsMenuOpen(false)}
            >
              Home
            </Link>
            <Link
              to="features"
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
              className="block px-3 py-2 text-text-primary hover:text-primary hover:bg-background-card rounded-md transition-colors cursor-pointer"
              onClick={() => setIsMenuOpen(false)}
            >
              Features
            </Link>
            <Link
              to="postures"
              spy={true}
              smooth={true}
              offset={-70}
              duration={500}
              className="block px-3 py-2 text-primary hover:bg-primary hover:text-text-primary rounded-md transition-colors cursor-pointer"
              onClick={() => setIsMenuOpen(false)}
            >
              Posture Check
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}

export default Navbar;