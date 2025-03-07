import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

const ModelsSection = styled.section`
  padding: 6rem 2rem;
  background: linear-gradient(135deg, var(--cream) 0%, var(--cream-dark) 100%);
  position: relative;
  overflow: hidden;
  
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

const BackgroundDecoration = styled.div`
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  background-image: 
    radial-gradient(circle at 10% 20%, rgba(139, 93, 51, 0.05) 0%, transparent 200px),
    radial-gradient(circle at 90% 80%, rgba(139, 93, 51, 0.05) 0%, transparent 200px);
  z-index: 0;
`;

const ModelsContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
`;

const SectionTitle = styled(motion.h2)`
  font-size: 3rem;
  text-align: center;
  margin-bottom: 1rem;
  color: var(--primary-color);
  position: relative;
`;

const SectionSubtitle = styled(motion.p)`
  text-align: center;
  max-width: 700px;
  margin: 0 auto 3rem;
  font-size: 1.2rem;
  color: var(--text-color);
  opacity: 0.8;
`;

const ModelsGrid = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ModelCard = styled(motion.div)`
  background-color: #fff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
  transition: transform var(--transition-medium), box-shadow var(--transition-medium);
  position: relative;
  
  &:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  }
`;

const ModelHeader = styled.div`
  background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
  color: #fff;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
  
  &:before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(255, 255, 255, 0.2) 0%, transparent 60%);
    opacity: 0;
    transform: scale(0.5);
    transition: opacity 0.6s ease, transform 0.6s ease;
  }
  
  ${ModelCard}:hover &:before {
    opacity: 1;
    transform: scale(1);
  }
`;

const ModelName = styled.h3`
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
  position: relative;
`;

const ModelDescription = styled.p`
  font-size: 0.95rem;
  opacity: 0.9;
  line-height: 1.5;
  position: relative;
`;

const ModelGrade = styled(motion.div)`
  position: absolute;
  top: 1.5rem;
  right: 1.5rem;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: ${({ grade }) => {
    if (grade === 'A+') return '#4CAF50';
    if (grade === 'A') return '#8BC34A';
    if (grade === 'C+') return '#FFC107';
    return '#F44336';
  }};
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 1rem;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
  z-index: 2;
`;

const ModelBody = styled.div`
  padding: 1.5rem;
`;

const ModelStat = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 1rem;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const StatLabel = styled.span`
  font-weight: 500;
  color: var(--text-color);
`;

const StatValue = styled.span`
  font-weight: 600;
  color: var(--primary-color);
`;

const StatBar = styled(motion.div)`
  height: 6px;
  background-color: #e0e0e0;
  border-radius: 3px;
  margin-top: 0.5rem;
  position: relative;
  overflow: hidden;
  
  &:after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 0;
    background-color: ${({ category }) => {
      if (category === 'knowledge') return '#4a6670';
      if (category === 'ethics') return '#8b5d33';
      if (category === 'bias') return '#6b8e23';
      return '#b38b6d';
    }};
    border-radius: 3px;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
  }
  
  &.animate:after {
    width: ${({ value }) => `${value}%`};
  }
`;

const FilterContainer = styled(motion.div)`
  display: flex;
  justify-content: center;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  gap: 0.5rem;
`;

const FilterButton = styled(motion.button)`
  padding: 0.5rem 1.5rem;
  background-color: ${({ active }) => active ? 'var(--primary-color)' : 'transparent'};
  color: ${({ active }) => active ? '#fff' : 'var(--primary-color)'};
  border: 2px solid var(--primary-color);
  border-radius: 30px;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-fast);
  
  &:hover {
    background-color: ${({ active }) => active ? 'var(--primary-color)' : 'rgba(74, 102, 112, 0.1)'};
    transform: translateY(-2px);
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

const Models = () => {
  const [ref, inView] = useInView({
    triggerOnce: false,
    threshold: 0.1,
  });
  
  const [filter, setFilter] = useState('all');
  const [hoveredCard, setHoveredCard] = useState(null);
  
  // Data from the evaluation results image
  const modelsData = [
    {
      name: 'GPT-4o',
      grade: 'A+',
      knowledge: 99.33,
      ethics: 97.50,
      bias: 96.00,
      source: 93.75,
      description: 'OpenAI\'s most advanced model, showing exceptional performance across all evaluation metrics.',
      category: 'openai'
    },
    {
      name: 'GPT-4.5 Preview',
      grade: 'A+',
      knowledge: 98.33,
      ethics: 97.50,
      bias: 96.00,
      source: 95.83,
      description: 'Preview version of GPT-4.5 with strong performance in source reliability and knowledge accuracy.',
      category: 'openai'
    },
    {
      name: 'Claude 3.5 - Sonnet',
      grade: 'A+',
      knowledge: 97.67,
      ethics: 100.00,
      bias: 98.00,
      source: 91.67,
      description: 'Anthropic\'s Claude 3.5 Sonnet model excels in ethical understanding and bias detection.',
      category: 'anthropic'
    },
    {
      name: 'Claude 3.7 - Sonnet',
      grade: 'A+',
      knowledge: 97.67,
      ethics: 97.50,
      bias: 96.00,
      source: 95.83,
      description: 'The latest Claude Sonnet model with improved source reliability compared to earlier versions.',
      category: 'anthropic'
    },
    {
      name: 'Claude 3.5 - Opus',
      grade: 'A+',
      knowledge: 97.33,
      ethics: 100.00,
      bias: 98.00,
      source: 93.75,
      description: 'Anthropic\'s larger Opus model with perfect scores in ethical understanding.',
      category: 'anthropic'
    },
    {
      name: 'Claude 3 - Opus',
      grade: 'A',
      knowledge: 95.67,
      ethics: 100.00,
      bias: 98.00,
      source: 91.67,
      description: 'Earlier version of Claude Opus with strong ethical reasoning capabilities.',
      category: 'anthropic'
    },
    {
      name: 'Gemini 2.0 - Flash',
      grade: 'A',
      knowledge: 97.00,
      ethics: 95.00,
      bias: 98.00,
      source: 91.67,
      description: 'Google\'s Gemini 2.0 Flash model with excellent bias detection and good knowledge accuracy.',
      category: 'google'
    },
    {
      name: 'Gemini Flash - 1.5',
      grade: 'C+',
      knowledge: 73.67,
      ethics: 92.50,
      bias: 96.00,
      source: 81.25,
      description: 'Earlier version of Gemini Flash with significantly lower knowledge accuracy scores.',
      category: 'google'
    },
    {
      name: 'GPT-4 Turbo',
      grade: 'C',
      knowledge: 93.67,
      ethics: 0.00,
      bias: 0.00,
      source: 93.75,
      description: 'GPT-4 Turbo shows good knowledge and source reliability but fails completely on ethics and bias metrics.',
      category: 'openai'
    }
  ];
  
  const filteredModels = filter === 'all' 
    ? modelsData 
    : modelsData.filter(model => model.category === filter);
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 0.5,
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] }
    }
  };
  
  const statBarVariants = {
    hidden: { width: 0 },
    visible: { 
      width: '100%',
      transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] }
    }
  };
  
  return (
    <ModelsSection id="models" ref={ref}>
      <BackgroundDecoration />
      <ModelsContainer>
        <SectionTitle
          variants={itemVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          Evaluated Models
        </SectionTitle>
        
        <SectionSubtitle
          variants={itemVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          Explore the performance of different AI language models on Islamic knowledge and ethical understanding
        </SectionSubtitle>
        
        <FilterContainer
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          <FilterButton 
            active={filter === 'all'} 
            onClick={() => setFilter('all')}
            variants={itemVariants}
            whileHover={{ y: -3 }}
            whileTap={{ scale: 0.95 }}
          >
            All Models
          </FilterButton>
          <FilterButton 
            active={filter === 'openai'} 
            onClick={() => setFilter('openai')}
            variants={itemVariants}
            whileHover={{ y: -3 }}
            whileTap={{ scale: 0.95 }}
          >
            OpenAI
          </FilterButton>
          <FilterButton 
            active={filter === 'anthropic'} 
            onClick={() => setFilter('anthropic')}
            variants={itemVariants}
            whileHover={{ y: -3 }}
            whileTap={{ scale: 0.95 }}
          >
            Anthropic
          </FilterButton>
          <FilterButton 
            active={filter === 'google'} 
            onClick={() => setFilter('google')}
            variants={itemVariants}
            whileHover={{ y: -3 }}
            whileTap={{ scale: 0.95 }}
          >
            Google
          </FilterButton>
        </FilterContainer>
        
        <ModelsGrid
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          {filteredModels.map((model, index) => (
            <ModelCard
              key={model.name}
              variants={itemVariants}
              onMouseEnter={() => setHoveredCard(model.name)}
              onMouseLeave={() => setHoveredCard(null)}
              whileHover={{ scale: 1.02 }}
            >
              <ModelHeader>
                <ModelName>{model.name}</ModelName>
                <ModelDescription>{model.description}</ModelDescription>
                <ModelGrade 
                  grade={model.grade}
                  animate={hoveredCard === model.name ? { scale: 1.1 } : { scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {model.grade}
                </ModelGrade>
              </ModelHeader>
              
              <ModelBody>
                <ModelStat>
                  <StatLabel>Knowledge Accuracy</StatLabel>
                  <StatValue>{model.knowledge}%</StatValue>
                </ModelStat>
                <StatBar 
                  value={model.knowledge} 
                  category="knowledge"
                  className={inView && hoveredCard === model.name ? 'animate' : ''}
                />
                
                <ModelStat>
                  <StatLabel>Ethics</StatLabel>
                  <StatValue>{model.ethics}%</StatValue>
                </ModelStat>
                <StatBar 
                  value={model.ethics} 
                  category="ethics"
                  className={inView && hoveredCard === model.name ? 'animate' : ''}
                />
                
                <ModelStat>
                  <StatLabel>Bias</StatLabel>
                  <StatValue>{model.bias}%</StatValue>
                </ModelStat>
                <StatBar 
                  value={model.bias} 
                  category="bias"
                  className={inView && hoveredCard === model.name ? 'animate' : ''}
                />
                
                <ModelStat>
                  <StatLabel>Source Reliability</StatLabel>
                  <StatValue>{model.source}%</StatValue>
                </ModelStat>
                <StatBar 
                  value={model.source} 
                  category="source"
                  className={inView && hoveredCard === model.name ? 'animate' : ''}
                />
              </ModelBody>
            </ModelCard>
          ))}
        </ModelsGrid>
        
        <NavigationButton 
          href="#"
          whileHover={{ y: -5 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 0.7 }}
        >
          ↑
        </NavigationButton>
      </ModelsContainer>
    </ModelsSection>
  );
};

export default Models; 