import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, 
  PolarRadiusAxis, Radar, Cell, LabelList
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
  margin-top: 3rem;
  height: 500px;
  position: relative;
  
  &:before {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.05);
    z-index: -1;
  }
`;

const TabsContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 2rem;
`;

const Tab = styled(motion.button)`
  padding: 0.8rem 1.5rem;
  background-color: ${({ active }) => active ? 'var(--primary-color)' : 'transparent'};
  color: ${({ active }) => active ? '#fff' : 'var(--primary-color)'};
  border: 2px solid var(--primary-color);
  border-radius: 4px;
  margin: 0 0.5rem;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-medium);
  position: relative;
  overflow: hidden;
  
  &:after {
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
    background-color: ${({ active }) => active ? 'var(--primary-color)' : 'rgba(74, 102, 112, 0.1)'};
    transform: translateY(-2px);
    
    &:after {
      transform: translateX(100%);
    }
  }
`;

const ModelSelector = styled(motion.div)`
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 2rem;
`;

const ModelButton = styled(motion.button)`
  padding: 0.5rem 1rem;
  background-color: ${({ active, color }) => active ? color : 'transparent'};
  color: ${({ active }) => active ? '#fff' : 'var(--text-color)'};
  border: 1px solid ${({ color }) => color};
  border-radius: 20px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all var(--transition-fast);
  
  &:hover {
    background-color: ${({ active, color }) => active ? color : `${color}33`};
    transform: translateY(-2px);
  }
`;

const CustomTooltip = styled.div`
  background-color: rgba(255, 255, 255, 0.95);
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 1rem;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
  
  .label {
    font-family: var(--title-font);
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--primary-color);
    font-size: 1.1rem;
  }
  
  .value {
    color: var(--accent-color);
    font-weight: 500;
    display: flex;
    align-items: center;
    margin-bottom: 0.3rem;
    
    &:before {
      content: '';
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      background-color: currentColor;
    }
  }
`;

const GradeIndicator = styled(motion.div)`
  position: absolute;
  top: 20px;
  right: 20px;
  background-color: ${({ grade }) => {
    if (grade === 'A+') return '#4CAF50';
    if (grade === 'A') return '#8BC34A';
    if (grade === 'C+') return '#FFC107';
    return '#F44336';
  }};
  color: white;
  font-weight: 700;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
  z-index: 2;
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

const Evaluation = () => {
  const [ref, inView] = useInView({
    triggerOnce: false,
    threshold: 0.1,
  });
  
  const [activeTab, setActiveTab] = useState('bar');
  const [selectedModels, setSelectedModels] = useState([
    'GPT-4o', 
    'Claude 3.5 - Sonnet', 
    'Claude 3.7 - Sonnet', 
    'Gemini 2.0 - Flash'
  ]);
  const [hoveredModel, setHoveredModel] = useState(null);
  const [activeCategory, setActiveCategory] = useState('all');
  
  // Vibrant colors for each model
  const modelColors = {
    'GPT-4o': '#FF5252',                  // Vibrant Red
    'GPT-4.5 Preview': '#FF7043',         // Deep Orange
    'Claude 3.5 - Sonnet': '#536DFE',     // Indigo
    'Claude 3.7 - Sonnet': '#448AFF',     // Blue
    'Claude 3.5 - Opus': '#40C4FF',       // Light Blue
    'Claude 3 - Opus': '#18FFFF',         // Cyan
    'Gemini 2.0 - Flash': '#69F0AE',      // Green
    'Gemini Flash - 1.5': '#B2FF59',      // Light Green
    'GPT-4 Turbo': '#FFFF00'              // Yellow
  };
  
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
  
  const filteredData = evaluationData.filter(model => 
    selectedModels.includes(model.name)
  );
  
  // Prepare data for the bar chart
  const prepareBarChartData = () => {
    if (activeCategory === 'all') {
      // For all categories, create a data structure that works with a grouped bar chart
      return [
        {
          name: 'Knowledge',
          ...filteredData.reduce((acc, model) => {
            acc[model.name] = model.knowledge;
            return acc;
          }, {})
        },
        {
          name: 'Ethics',
          ...filteredData.reduce((acc, model) => {
            acc[model.name] = model.ethics;
            return acc;
          }, {})
        },
        {
          name: 'Bias',
          ...filteredData.reduce((acc, model) => {
            acc[model.name] = model.bias;
            return acc;
          }, {})
        },
        {
          name: 'Source',
          ...filteredData.reduce((acc, model) => {
            acc[model.name] = model.source;
            return acc;
          }, {})
        }
      ];
    } else {
      // For a single category, return data for that category only
      return filteredData.map(model => ({
        name: model.name,
        value: model[activeCategory],
        color: model.color
      }));
    }
  };
  
  const barChartData = prepareBarChartData();
  
  const toggleModel = (modelName) => {
    if (selectedModels.includes(modelName)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter(name => name !== modelName));
      }
    } else {
      setSelectedModels([...selectedModels, modelName]);
    }
  };
  
  const renderCustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      if (activeCategory === 'all') {
        return (
          <CustomTooltip>
            <p className="label">{label}</p>
            {payload.map((entry, index) => (
              <p key={index} className="value" style={{ color: entry.color }}>
                {`${entry.name}: ${entry.value}%`}
              </p>
            ))}
          </CustomTooltip>
        );
      } else {
        const model = evaluationData.find(m => m.name === label);
        return (
          <CustomTooltip>
            <p className="label">{label} <span style={{ fontSize: '0.8rem' }}>({model.grade})</span></p>
            <p className="value" style={{ color: model.color }}>
              {`${activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)}: ${payload[0].value}%`}
            </p>
          </CustomTooltip>
        );
      }
    }
    return null;
  };
  
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
  
  const chartVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: { 
        duration: 0.8, 
        ease: [0.16, 1, 0.3, 1],
        delay: 0.3
      }
    }
  };
  
  return (
    <EvaluationSection id="evaluation" ref={ref}>
      <BackgroundPattern />
      <EvaluationContainer>
        <SectionTitle
          variants={itemVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          Evaluation Results
        </SectionTitle>
        
        <SectionSubtitle
          variants={itemVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          Compare how different AI models perform on Islamic knowledge, ethics, bias detection, and source reliability
        </SectionSubtitle>
        
        <ModelSelector
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          {evaluationData.map((model) => (
            <ModelButton
              key={model.name}
              active={selectedModels.includes(model.name)}
              onClick={() => toggleModel(model.name)}
              variants={itemVariants}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
              whileTap={{ scale: 0.95 }}
              onMouseEnter={() => setHoveredModel(model.name)}
              onMouseLeave={() => setHoveredModel(null)}
              color={model.color}
            >
              {model.name}
              {hoveredModel === model.name && (
                <motion.span
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  style={{ 
                    marginLeft: '5px',
                    display: 'inline-block',
                    padding: '0 5px',
                    borderRadius: '50%',
                    backgroundColor: selectedModels.includes(model.name) ? 'rgba(255, 255, 255, 0.3)' : `${model.color}33`,
                    fontSize: '0.8rem'
                  }}
                >
                  {model.grade}
                </motion.span>
              )}
            </ModelButton>
          ))}
        </ModelSelector>
        
        <TabsContainer>
          <Tab 
            active={activeTab === 'bar'} 
            onClick={() => setActiveTab('bar')}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.95 }}
            variants={itemVariants}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
          >
            Bar Chart
          </Tab>
          <Tab 
            active={activeTab === 'radar'} 
            onClick={() => setActiveTab('radar')}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.95 }}
            variants={itemVariants}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
          >
            Radar Chart
          </Tab>
        </TabsContainer>
        
        {activeTab === 'bar' && (
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
        )}
        
        <AnimatePresence mode="wait">
          <ChartContainer
            key={activeTab + activeCategory}
            variants={chartVariants}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
            exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.3 } }}
          >
            {activeTab === 'bar' ? (
              <>
                <ChartTitle>
                  {activeCategory === 'all' 
                    ? 'Model Performance Across All Categories' 
                    : `Model Performance: ${activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)}`}
                </ChartTitle>
                <ResponsiveContainer width="100%" height="90%">
                  {activeCategory === 'all' ? (
                    <BarChart
                      data={barChartData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                      barGap={2}
                      barCategoryGap={10}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
                      <XAxis 
                        dataKey="name" 
                        tick={{ fontSize: 12, fill: 'var(--text-color)' }}
                        tickLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                        axisLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                      />
                      <YAxis 
                        domain={[0, 100]} 
                        label={{ 
                          value: 'Score (%)', 
                          angle: -90, 
                          position: 'insideLeft',
                          style: { textAnchor: 'middle', fill: 'var(--text-color)' }
                        }} 
                        tick={{ fill: 'var(--text-color)' }}
                        tickLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                        axisLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                      />
                      <Tooltip content={renderCustomTooltip} />
                      <Legend 
                        wrapperStyle={{ paddingTop: '20px' }}
                        formatter={(value) => (
                          <span style={{ 
                            color: modelColors[value] || 'var(--text-color)', 
                            fontWeight: 500 
                          }}>
                            {value}
                          </span>
                        )}
                      />
                      {filteredData.map((model) => (
                        <Bar 
                          key={model.name} 
                          dataKey={model.name} 
                          fill={model.color} 
                          name={model.name}
                          fillOpacity={hoveredModel === model.name ? 1 : 0.8}
                        />
                      ))}
                    </BarChart>
                  ) : (
                    <BarChart
                      data={barChartData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                      barSize={40}
                      layout="horizontal"
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={true} />
                      <XAxis 
                        dataKey="name" 
                        angle={-45} 
                        textAnchor="end" 
                        height={80} 
                        tick={{ fontSize: 12, fill: 'var(--text-color)' }}
                        tickLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                        axisLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                      />
                      <YAxis 
                        domain={[0, 100]} 
                        label={{ 
                          value: 'Score (%)', 
                          angle: -90, 
                          position: 'insideLeft',
                          style: { textAnchor: 'middle', fill: 'var(--text-color)' }
                        }} 
                        tick={{ fill: 'var(--text-color)' }}
                        tickLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                        axisLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                      />
                      <Tooltip content={renderCustomTooltip} />
                      <Bar 
                        dataKey="value" 
                        name={activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)}
                      >
                        <LabelList dataKey="value" position="top" formatter={(value) => `${value}%`} />
                        {barChartData.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.color}
                            fillOpacity={hoveredModel === entry.name ? 1 : 0.8}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  )}
                </ResponsiveContainer>
              </>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart outerRadius={180} data={filteredData}>
                  <PolarGrid stroke="rgba(0,0,0,0.05)" />
                  <PolarAngleAxis 
                    dataKey="name" 
                    tick={{ fontSize: 12, fill: 'var(--text-color)' }}
                  />
                  <PolarRadiusAxis 
                    domain={[0, 100]} 
                    tick={{ fill: 'var(--text-color)' }}
                    axisLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                    tickLine={{ stroke: 'rgba(0,0,0,0.05)' }}
                  />
                  <Tooltip content={renderCustomTooltip} />
                  {filteredData.map((model) => (
                    <Radar 
                      key={model.name}
                      name={model.name} 
                      dataKey={(entry) => (
                        (entry.knowledge + entry.ethics + entry.bias + entry.source) / 4
                      )}
                      stroke={model.color} 
                      fill={model.color} 
                      fillOpacity={0.5} 
                    />
                  ))}
                  <Legend 
                    wrapperStyle={{ paddingTop: '20px' }}
                    formatter={(value) => {
                      const model = evaluationData.find(m => m.name === value);
                      return (
                        <span style={{ 
                          color: model ? model.color : 'var(--text-color)', 
                          fontWeight: 500 
                        }}>
                          {value}
                        </span>
                      );
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            )}
            
            {filteredData.length === 1 && (
              <GradeIndicator
                grade={filteredData[0].grade}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.3 }}
              >
                {filteredData[0].grade}
              </GradeIndicator>
            )}
          </ChartContainer>
        </AnimatePresence>
        
        <NavigationButton 
          href="#models"
          whileHover={{ y: -5 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 0.7 }}
        >
          ↓
        </NavigationButton>
      </EvaluationContainer>
    </EvaluationSection>
  );
};

export default Evaluation; 