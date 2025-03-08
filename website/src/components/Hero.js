import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import mosque1 from '../assets/images/mosque1.jpg';

const HeroSection = styled.section`
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.4)), 
              url(${mosque1});
  background-size: cover;
  background-position: center;
  color: #fff;
  text-align: center;
  position: relative;
  overflow: hidden;
`;

const HeroOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at center, rgba(74, 102, 112, 0.2) 0%, rgba(74, 102, 112, 0.4) 100%);
  z-index: 1;
`;

const HeroContent = styled.div`
  max-width: 800px;
  padding: 0 2rem;
  z-index: 2;
  position: relative;
`;

const HeroTitle = styled(motion.h1)`
  font-size: 5.5rem;
  margin-bottom: 1.5rem;
  font-weight: 600;
  line-height: 1.1;
  letter-spacing: 0.02em;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
  
  @media (max-width: 768px) {
    font-size: 3.5rem;
  }
`;

const HeroSubtitle = styled(motion.p)`
  font-size: 1.5rem;
  margin-bottom: 2rem;
  opacity: 0.9;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
  
  @media (max-width: 768px) {
    font-size: 1.2rem;
  }
`;

const HeroButton = styled(motion.a)`
  display: inline-block;
  background-color: var(--secondary-color);
  color: var(--text-color);
  padding: 0.8rem 2rem;
  border-radius: 4px;
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  cursor: pointer;
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.2);
    transform: translateX(-100%);
    transition: transform 0.6s cubic-bezier(0.16, 1, 0.3, 1);
  }
  
  &:hover {
    background-color: #d8c9b0;
    transform: translateY(-2px);
    
    &:before {
      transform: translateX(100%);
    }
  }
`;

const Hero = () => {
  const titleVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.16, 1, 0.3, 1]
      }
    }
  };
  
  const subtitleVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        delay: 0.2,
        ease: [0.16, 1, 0.3, 1]
      }
    }
  };
  
  const buttonVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        delay: 0.4,
        ease: [0.16, 1, 0.3, 1]
      }
    },
    hover: {
      scale: 1.05,
      transition: {
        duration: 0.3,
        ease: [0.16, 1, 0.3, 1]
      }
    },
    tap: {
      scale: 0.95
    }
  };
  
  return (
    <HeroSection>
      <HeroOverlay />
      
      <HeroContent>
        <HeroTitle
          variants={titleVariants}
          initial="hidden"
          animate="visible"
        >
          Islamic GenAI Guild
        </HeroTitle>
        <HeroSubtitle
          variants={subtitleVariants}
          initial="hidden"
          animate="visible"
        >
          LLM Evaluation Initiative
        </HeroSubtitle>
        <HeroButton
          href="#about"
          variants={buttonVariants}
          initial="hidden"
          animate="visible"
          whileHover="hover"
          whileTap="tap"
        >
          Learn More
        </HeroButton>
      </HeroContent>
    </HeroSection>
  );
};

export default Hero; 