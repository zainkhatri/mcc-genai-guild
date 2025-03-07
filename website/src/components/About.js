import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import mosque2 from '../assets/images/mosque2.jpg';

const AboutSection = styled.section`
  padding: 6rem 2rem;
  background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream) 100%);
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 100px;
    background: linear-gradient(to bottom, rgba(248, 245, 240, 1), rgba(248, 245, 240, 0));
    z-index: 1;
  }
`;

const AboutContainer = styled.div`
  max-width: 1000px;
  margin: 0 auto;
  position: relative;
  z-index: 2;
`;

const SectionTitle = styled(motion.h2)`
  font-size: 2.5rem;
  text-align: center;
  margin-bottom: 3rem;
  color: var(--primary-color);
  position: relative;
  
  &:after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background-color: var(--secondary-color);
  }
`;

const AboutContent = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  align-items: center;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const AboutText = styled.div`
  p {
    margin-bottom: 1.5rem;
    font-size: 1.1rem;
    line-height: 1.8;
  }
`;

const AboutImage = styled(motion.div)`
  img {
    width: 100%;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  }
  
  @media (max-width: 768px) {
    grid-row: 1;
  }
`;

const NavigationButton = styled(motion.a)`
  display: block;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background-color: var(--primary-color);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 3rem auto 0;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  font-size: 1.5rem;
  
  &:hover {
    background-color: var(--accent-color);
  }
`;

const About = () => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });
  
  return (
    <AboutSection id="about" ref={ref}>
      <AboutContainer>
        <SectionTitle
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6 }}
        >
          About the Project
        </SectionTitle>
        
        <AboutContent>
          <AboutText>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              The <strong>Islamic LLM Evaluation Project</strong> is dedicated to benchmarking AI language models on their understanding of Islamic knowledge, ethical reasoning, and bias detection. Using the <code>lm-evaluation-harness</code>, we test models across structured Islamic Q&A, ethical scenarios, and source reliability.
            </motion.p>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              Our goal is to ensure that AI systems can accurately represent Islamic teachings and values, providing a valuable resource for researchers, developers, and the Muslim community. The project evaluates models on four key metrics: Islamic Knowledge Accuracy, Ethical Understanding, Bias Against Islam, and Source Reliability.
            </motion.p>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              Results are compiled into a comprehensive leaderboard that helps users identify which AI models best align with Islamic principles and accurately represent Islamic knowledge.
            </motion.p>
          </AboutText>
          
          <AboutImage
            initial={{ opacity: 0, scale: 0.9 }}
            animate={inView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <img 
              src={mosque2} 
              alt="Islamic architecture" 
            />
          </AboutImage>
        </AboutContent>
        
        <NavigationButton 
          href="#objectives"
          whileHover={{ y: -5 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 0.7 }}
        >
          ↓
        </NavigationButton>
      </AboutContainer>
    </AboutSection>
  );
};

export default About; 