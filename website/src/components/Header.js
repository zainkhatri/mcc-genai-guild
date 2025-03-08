import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

// Hidden header that still provides navigation functionality
const HeaderContainer = styled(motion.header)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  pointer-events: none; // Make the header container not block interactions
`;

const Nav = styled.nav`
  display: flex;
  justify-content: flex-end;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
  padding: 1.5rem 2rem;
`;

const MobileMenuButton = styled(motion.button)`
  display: block;
  background: rgba(74, 102, 112, 0.7);
  border: none;
  color: #fff;
  font-size: 1.5rem;
  cursor: pointer;
  z-index: 101;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
  pointer-events: auto; // Allow interaction with this button
  backdrop-filter: blur(5px);
  
  @media (min-width: 769px) {
    display: none; // Hide on desktop
  }
`;

const DesktopNav = styled.div`
  display: none;
  
  @media (min-width: 769px) {
    display: flex;
    gap: 2rem;
    background: rgba(74, 102, 112, 0.7);
    padding: 0.8rem 1.5rem;
    border-radius: 30px;
    backdrop-filter: blur(5px);
    pointer-events: auto; // Allow interaction
  }
`;

const NavLink = styled(motion.a)`
  color: #fff;
  font-weight: 500;
  text-decoration: none;
  position: relative;
  padding: 0.5rem 0;
  
  &:after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 0;
    height: 2px;
    background-color: #fff;
    transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  }
  
  &:hover {
    &:after {
      width: 100%;
    }
  }
`;

const MobileMenu = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(74, 102, 112, 0.98);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 100;
  padding: 2rem;
`;

const MobileNavLinks = styled(motion.ul)`
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 2rem;
  text-align: center;
`;

const MobileNavLink = styled(motion.li)`
  a {
    color: #fff;
    font-size: 2rem;
    font-weight: 600;
    font-family: var(--title-font);
    transition: color 0.3s ease;
    
    &:hover {
      color: var(--secondary-color);
    }
  }
`;

const Header = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeLink, setActiveLink] = useState(null);
  
  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
    
    // Prevent scrolling when mobile menu is open
    if (!mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }
  };
  
  const closeMobileMenu = () => {
    setMobileMenuOpen(false);
    document.body.style.overflow = 'auto';
  };
  
  const navLinks = [
    { id: 'about', label: 'About' },
    { id: 'objectives', label: 'Objectives' },
    { id: 'models', label: 'Models' },
    { id: 'evaluation', label: 'Evaluation' },
    { id: 'ethics', label: 'Ethics' }
  ];
  
  const headerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 0.6, 
        ease: [0.16, 1, 0.3, 1] 
      }
    }
  };
  
  const linkVariants = {
    hover: { 
      scale: 1.05,
      transition: { duration: 0.2 }
    },
    tap: { scale: 0.95 }
  };
  
  const mobileMenuVariants = {
    closed: { 
      opacity: 0,
      y: -20,
      transition: {
        duration: 0.5,
        ease: [0.16, 1, 0.3, 1],
        staggerChildren: 0.1,
        staggerDirection: -1
      }
    },
    open: { 
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: [0.16, 1, 0.3, 1],
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };
  
  const mobileLinkVariants = {
    closed: { opacity: 0, y: 20 },
    open: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] }
    }
  };
  
  return (
    <HeaderContainer 
      variants={headerVariants}
      initial="hidden"
      animate="visible"
    >
      <Nav>
        <DesktopNav>
          {navLinks.map((link) => (
            <NavLink 
              key={link.id}
              href={`#${link.id}`}
              variants={linkVariants}
              whileHover="hover"
              whileTap="tap"
              onMouseEnter={() => setActiveLink(link.id)}
              onMouseLeave={() => setActiveLink(null)}
            >
              {link.label}
            </NavLink>
          ))}
        </DesktopNav>
        
        <MobileMenuButton 
          onClick={toggleMobileMenu}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          {mobileMenuOpen ? '✕' : '☰'}
        </MobileMenuButton>
      </Nav>
      
      <AnimatePresence>
        {mobileMenuOpen && (
          <MobileMenu
            initial="closed"
            animate="open"
            exit="closed"
            variants={mobileMenuVariants}
          >
            <MobileNavLinks>
              {navLinks.map((link) => (
                <MobileNavLink
                  key={link.id}
                  variants={mobileLinkVariants}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <a 
                    href={`#${link.id}`}
                    onClick={closeMobileMenu}
                  >
                    {link.label}
                  </a>
                </MobileNavLink>
              ))}
            </MobileNavLinks>
          </MobileMenu>
        )}
      </AnimatePresence>
    </HeaderContainer>
  );
};

export default Header; 