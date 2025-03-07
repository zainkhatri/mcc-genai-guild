import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

const ObjectivesSection = styled.section`
  padding: 6rem 2rem;
  background: linear-gradient(135deg, var(--cream) 0%, var(--cream-dark) 100%);
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 100px;
    background: linear-gradient(to bottom, rgba(240, 230, 210, 1), rgba(240, 230, 210, 0));
    z-index: 1;
  }
`;

const ObjectivesContainer = styled.div`
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

const ObjectivesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ObjectiveCard = styled(motion.div)`
  background-color: #fff;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
  }
`;

const ObjectiveIcon = styled.div`
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: var(--accent-color);
`;

const ObjectiveTitle = styled.h3`
  font-size: 1.5rem;
  margin-bottom: 1rem;
  color: var(--primary-color);
`;

const ObjectiveDescription = styled.p`
  font-size: 1rem;
  line-height: 1.6;
  color: var(--text-color);
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

const Objectives = () => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });
  
  const objectives = [
    {
      icon: '📚',
      title: 'Comprehensive Datasets',
      description: 'Develop extensive datasets covering Islamic knowledge, ethics, and cultural contexts to ensure thorough evaluation of language models.'
    },
    {
      icon: '🔍',
      title: 'Evaluation Framework',
      description: 'Create robust frameworks to assess how well AI models understand and represent Islamic concepts, teachings, and values.'
    },
    {
      icon: '📊',
      title: 'Grading System',
      description: 'Implement a transparent grading system that evaluates models on knowledge accuracy, ethical understanding, bias detection, and source reliability.'
    },
    {
      icon: '🏆',
      title: 'Leaderboard Maintenance',
      description: 'Maintain up-to-date leaderboards that help users identify which AI models best align with Islamic principles and accurately represent Islamic knowledge.'
    }
  ];
  
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: 0.2 + i * 0.1,
        duration: 0.5,
      }
    })
  };
  
  return (
    <ObjectivesSection id="objectives" ref={ref}>
      <ObjectivesContainer>
        <SectionTitle
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6 }}
        >
          Our Objectives
        </SectionTitle>
        
        <ObjectivesGrid>
          {objectives.map((objective, index) => (
            <ObjectiveCard
              key={index}
              custom={index}
              initial="hidden"
              animate={inView ? "visible" : "hidden"}
              variants={cardVariants}
            >
              <ObjectiveIcon>{objective.icon}</ObjectiveIcon>
              <ObjectiveTitle>{objective.title}</ObjectiveTitle>
              <ObjectiveDescription>{objective.description}</ObjectiveDescription>
            </ObjectiveCard>
          ))}
        </ObjectivesGrid>
        
        <NavigationButton 
          href="#evaluation"
          whileHover={{ y: -5 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 0.7 }}
        >
          ↓
        </NavigationButton>
      </ObjectivesContainer>
    </ObjectivesSection>
  );
};

export default Objectives; 