import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Cell
} from 'recharts';

const EvaluationSection = styled.section`
  padding: 6rem 2rem;
  background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream) 100%);
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

const BackgroundPattern = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image: radial-gradient(rgba(139, 93, 51, 0.05) 1px, transparent 1px);
  background-size: 20px 20px;
  opacity: 0.5;
  z-index: 0;
`;

const EvaluationContainer = styled.div`
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

const ChartContainer = styled(motion.div)`
  background-color: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  height: 600px;
  position: relative;
  overflow: visible;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(139, 93, 51, 0.1);
  margin-top: 2rem;
  
  /* Add a subtle grid pattern to the background */
  background-image: 
    linear-gradient(rgba(139, 93, 51, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(139, 93, 51, 0.03) 1px, transparent 1px);
  background-size: 20px 20px;
`;

const CategorySelector = styled.div`
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 2rem;
`;

const CategoryButton = styled.button`
  padding: 0.5rem 1rem;
  background-color: ${({ active }) => active ? 'var(--primary-color)' : 'rgba(74, 102, 112, 0.1)'};
  color: ${({ active }) => active ? '#fff' : 'var(--text-color)'};
  border: none;
  border-radius: 20px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background-color: ${({ active }) => active ? 'var(--primary-color)' : 'rgba(74, 102, 112, 0.2)'};
  }
`;

const ChartTitle = styled.h3`
  text-align: center;
  font-size: 1.5rem;
  margin-bottom: 1.5rem;
  color: var(--primary-color);
`;

const CustomTooltip = styled.div`
  background-color: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(139, 93, 51, 0.2);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(4px);
  
  .label {
    margin: 0 0 8px;
    font-weight: 600;
    color: var(--primary-color);
  }
  
  .value {
    margin: 4px 0;
    font-size: 0.9rem;
    color: var(--text-color);
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

const ModelGroupTitle = styled(motion.div)`
  text-align: center;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  color: var(--primary-color);
  font-weight: 500;
  
  span {
    display: inline-block;
    margin: 0 0.3rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
  }
`;

const Evaluation = () => {
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: false
  });
  
  const [activeCategory, setActiveCategory] = useState('all');
  
  // Update the modelColors to use colors that match the website's aesthetic
  const modelColors = {
    'GPT-4o': '#8b5d33',                  // Primary brown from the website
    'GPT-4.5 Preview': '#a06e44',         // Lighter brown
    'Claude 3.5 - Sonnet': '#4a6670',     // Primary blue-gray from the website
    'Claude 3.7 - Sonnet': '#5a7680',     // Lighter blue-gray
    'Claude 3.5 - Opus': '#6a8690',       // Even lighter blue-gray
    'Claude 3 - Opus': '#7a96a0',         // Lightest blue-gray
    'Gemini 2.0 - Flash': '#b07d4a',      // Warm gold
    'Gemini Flash - 1.5': '#c99c6e',      // Light gold
    'GPT-4 Turbo': '#d4b28c'              // Cream accent
  };
  
  // Use useMemo to prevent the modelGroups array from changing on every render
  const modelGroups = React.useMemo(() => [
    // Group 1: Top performers
    ['GPT-4o', 'Claude 3.5 - Sonnet', 'Claude 3.7 - Sonnet', 'GPT-4.5 Preview'],
    // Group 2: Strong performers
    ['Claude 3.5 - Opus', 'Claude 3 - Opus', 'Gemini 2.0 - Flash', 'GPT-4 Turbo'],
    // Group 3: Mixed performance
    ['GPT-4o', 'Claude 3.5 - Sonnet', 'Gemini 2.0 - Flash', 'Gemini Flash - 1.5'],
    // Group 4: Compare OpenAI models
    ['GPT-4o', 'GPT-4.5 Preview', 'GPT-4 Turbo', 'Claude 3.5 - Sonnet']
  ], []);
  
  // State for currently displayed models
  const [selectedModels, setSelectedModels] = useState(modelGroups[0]);
  const [currentGroupIndex, setCurrentGroupIndex] = useState(0);
  
  // Data from the evaluation results image
  const evaluationData = [
    {
      name: 'GPT-4o',
      grade: 'A+',
      knowledge: 99.33,
      ethics: 97.50,
      bias: 96.00,
      source: 93.75,
      color: modelColors['GPT-4o']
    },
    {
      name: 'GPT-4.5 Preview',
      grade: 'A+',
      knowledge: 98.33,
      ethics: 97.50,
      bias: 96.00,
      source: 95.83,
      color: modelColors['GPT-4.5 Preview']
    },
    {
      name: 'Claude 3.5 - Sonnet',
      grade: 'A+',
      knowledge: 97.67,
      ethics: 100.00,
      bias: 98.00,
      source: 91.67,
      color: modelColors['Claude 3.5 - Sonnet']
    },
    {
      name: 'Claude 3.7 - Sonnet',
      grade: 'A+',
      knowledge: 97.67,
      ethics: 97.50,
      bias: 96.00,
      source: 95.83,
      color: modelColors['Claude 3.7 - Sonnet']
    },
    {
      name: 'Claude 3.5 - Opus',
      grade: 'A+',
      knowledge: 97.33,
      ethics: 100.00,
      bias: 98.00,
      source: 93.75,
      color: modelColors['Claude 3.5 - Opus']
    },
    {
      name: 'Claude 3 - Opus',
      grade: 'A',
      knowledge: 95.67,
      ethics: 100.00,
      bias: 98.00,
      source: 91.67,
      color: modelColors['Claude 3 - Opus']
    },
    {
      name: 'Gemini 2.0 - Flash',
      grade: 'A',
      knowledge: 97.00,
      ethics: 95.00,
      bias: 98.00,
      source: 91.67,
      color: modelColors['Gemini 2.0 - Flash']
    },
    {
      name: 'Gemini Flash - 1.5',
      grade: 'C+',
      knowledge: 73.67,
      ethics: 92.50,
      bias: 96.00,
      source: 81.25,
      color: modelColors['Gemini Flash - 1.5']
    },
    {
      name: 'GPT-4 Turbo',
      grade: 'C',
      knowledge: 93.67,
      ethics: 0.00,
      bias: 0.00,
      source: 93.75,
      color: modelColors['GPT-4 Turbo']
    }
  ];
  
  // Update the useEffect to prevent shuffling
  useEffect(() => {
    // Remove the shuffling interval to keep colors consistent
    // This will make the chart display only the first group of models
    if (!inView) return;
    
    // Set to the first group and don't change
    setCurrentGroupIndex(0);
    setSelectedModels(modelGroups[0]);
    
  }, [inView, modelGroups]);
  
  // Prepare data for the bar chart
  const prepareBarChartData = () => {
    if (activeCategory === 'all') {
      // For all categories, create a data structure that works with a grouped bar chart
      return [
        {
          name: 'Knowledge',
          ...selectedModels.reduce((acc, modelName) => {
            const model = evaluationData.find(m => m.name === modelName);
            acc[modelName] = model.knowledge;
            return acc;
          }, {})
        },
        {
          name: 'Ethics',
          ...selectedModels.reduce((acc, modelName) => {
            const model = evaluationData.find(m => m.name === modelName);
            acc[modelName] = model.ethics;
            return acc;
          }, {})
        },
        {
          name: 'Bias',
          ...selectedModels.reduce((acc, modelName) => {
            const model = evaluationData.find(m => m.name === modelName);
            acc[modelName] = model.bias;
            return acc;
          }, {})
        },
        {
          name: 'Source',
          ...selectedModels.reduce((acc, modelName) => {
            const model = evaluationData.find(m => m.name === modelName);
            acc[modelName] = model.source;
            return acc;
          }, {})
        }
      ];
    } else {
      // For a single category, return data for that category only
      return selectedModels.map(modelName => ({
        name: modelName,
        value: evaluationData.find(m => m.name === modelName)[activeCategory],
        color: modelColors[modelName]
      }));
    }
  };
  
  const barChartData = prepareBarChartData();
  
  const chartVariants = {
    hidden: { 
      opacity: 0,
      scale: 0.95,
      transition: { 
        duration: 0.3,
        ease: [0.43, 0.13, 0.23, 0.96]
      }
    },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: { 
        duration: 0.5,
        ease: [0.43, 0.13, 0.23, 0.96]
      }
    }
  };
  
  // Update the renderCustomTooltip function
  const renderCustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <CustomTooltip>
          <p className="label">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="value" style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value.toFixed(2)}%`}
            </p>
          ))}
        </CustomTooltip>
      );
    }
    return null;
  };
  
  // Add the filteredData definition
  const filteredData = selectedModels.map(modelName => 
    evaluationData.find(model => model.name === modelName)
  );
  
  return (
    <EvaluationSection id="evaluation" ref={ref}>
      <BackgroundPattern />
      <EvaluationContainer>
        <SectionTitle
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          Evaluation Results
        </SectionTitle>
        
        <SectionSubtitle
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
        >
          Comprehensive analysis of leading AI models based on knowledge, ethics, bias, and source attribution.
        </SectionSubtitle>
        
        <ModelGroupTitle
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
        >
          Comparing: {selectedModels.join(', ')}
        </ModelGroupTitle>
        
        <AnimatePresence mode="wait">
          <ChartContainer
            key={activeCategory + currentGroupIndex}
            variants={chartVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
          >
            <CategorySelector>
              <CategoryButton 
                active={activeCategory === 'all'} 
                onClick={() => setActiveCategory('all')}
              >
                All Categories
              </CategoryButton>
              <CategoryButton 
                active={activeCategory === 'knowledge'} 
                onClick={() => setActiveCategory('knowledge')}
              >
                Knowledge
              </CategoryButton>
              <CategoryButton 
                active={activeCategory === 'ethics'} 
                onClick={() => setActiveCategory('ethics')}
              >
                Ethics
              </CategoryButton>
              <CategoryButton 
                active={activeCategory === 'bias'} 
                onClick={() => setActiveCategory('bias')}
              >
                Bias
              </CategoryButton>
              <CategoryButton 
                active={activeCategory === 'source'} 
                onClick={() => setActiveCategory('source')}
              >
                Source
              </CategoryButton>
            </CategorySelector>
            
            <ChartTitle>
              {activeCategory === 'all' 
                ? 'Performance Across All Categories' 
                : `Performance in ${activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)}`}
            </ChartTitle>
            
            <ResponsiveContainer width="100%" height="80%">
              {activeCategory === 'all' ? (
                <BarChart data={barChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 93, 51, 0.1)" />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fill: 'var(--text-color)' }}
                    axisLine={{ stroke: 'rgba(139, 93, 51, 0.3)' }}
                  />
                  <YAxis 
                    domain={[0, 100]} 
                    tick={{ fill: 'var(--text-color)' }}
                    axisLine={{ stroke: 'rgba(139, 93, 51, 0.3)' }}
                    tickLine={{ stroke: 'rgba(139, 93, 51, 0.3)' }}
                  />
                  <Tooltip content={renderCustomTooltip} />
                  <Legend />
                  {selectedModels.map((modelName, index) => {
                    const model = evaluationData.find(m => m.name === modelName);
                    return (
                      <Bar 
                        key={modelName} 
                        dataKey={modelName} 
                        fill={model.color}
                        radius={[4, 4, 0, 0]}
                      />
                    );
                  })}
                </BarChart>
              ) : (
                <BarChart data={barChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 93, 51, 0.1)" />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fill: 'var(--text-color)' }}
                    axisLine={{ stroke: 'rgba(139, 93, 51, 0.3)' }}
                  />
                  <YAxis 
                    domain={[0, 100]} 
                    tick={{ fill: 'var(--text-color)' }}
                    axisLine={{ stroke: 'rgba(139, 93, 51, 0.3)' }}
                    tickLine={{ stroke: 'rgba(139, 93, 51, 0.3)' }}
                  />
                  <Tooltip content={renderCustomTooltip} />
                  <Bar 
                    dataKey="value" 
                    radius={[4, 4, 0, 0]}
                  >
                    {barChartData.map((entry, index) => {
                      const model = evaluationData.find(m => m.name === entry.name);
                      return (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={model ? model.color : '#ccc'}
                        />
                      );
                    })}
                  </Bar>
                </BarChart>
              )}
            </ResponsiveContainer>
          </ChartContainer>
        </AnimatePresence>
        
        <NavigationButton 
          href="#technical"
          whileHover={{ y: -5 }}
          whileTap={{ y: 0 }}
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
        >
          Explore Technical Details
        </NavigationButton>
      </EvaluationContainer>
    </EvaluationSection>
  );
};

export default Evaluation; 