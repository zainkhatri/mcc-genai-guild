import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

const TechnicalSection = styled.section`
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
    background: linear-gradient(to bottom, rgba(240, 230, 210, 1), rgba(240, 230, 210, 0));
    z-index: 1;
  }
`;

const TechnicalContainer = styled.div`
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

const AgentContainer = styled.div`
  margin-bottom: 4rem;
`;

const AgentTitle = styled(motion.h3)`
  font-size: 1.8rem;
  color: var(--primary-color);
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  
  &:before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 30px;
    background-color: var(--accent-color);
    margin-right: 15px;
    border-radius: 4px;
  }
`;

const AgentDescription = styled(motion.div)`
  background-color: #fff;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
  margin-bottom: 2rem;
  
  p {
    margin-bottom: 1rem;
    line-height: 1.8;
    font-size: 1.1rem;
  }
  
  ul {
    margin-left: 1.5rem;
    margin-bottom: 1.5rem;
    
    li {
      margin-bottom: 0.5rem;
      line-height: 1.6;
    }
  }
`;

const AgentFeatures = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
`;

const FeatureCard = styled(motion.div)`
  background-color: rgba(255, 255, 255, 0.7);
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
  
  h4 {
    color: var(--accent-color);
    margin-bottom: 1rem;
    font-size: 1.2rem;
  }
  
  p {
    font-size: 1rem;
    line-height: 1.6;
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

const TechnicalComponents = () => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 0.5,
        staggerChildren: 0.2
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
  
  return (
    <TechnicalSection id="technical" ref={ref}>
      <TechnicalContainer>
        <SectionTitle
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6 }}
        >
          Technical Components
        </SectionTitle>
        
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          <AgentContainer>
            <AgentTitle variants={itemVariants}>Agent 1A: Data Collection & Curation</AgentTitle>
            <AgentDescription variants={itemVariants}>
              <p>
                Agent 1A is responsible for gathering and curating high-quality Islamic content from authoritative sources. This agent ensures that our evaluation dataset is comprehensive, accurate, and representative of Islamic knowledge.
              </p>
              <p>Key responsibilities include:</p>
              <ul>
                <li>Collecting Quranic verses, authentic Hadiths, and scholarly interpretations</li>
                <li>Verifying source authenticity and reliability</li>
                <li>Categorizing content by topic, complexity, and relevance</li>
                <li>Ensuring balanced representation across different Islamic schools of thought</li>
              </ul>
              
              <AgentFeatures>
                <FeatureCard variants={itemVariants}>
                  <h4>Source Verification</h4>
                  <p>Implements rigorous verification protocols to ensure all content comes from recognized Islamic authorities and scholarly sources.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Metadata Tagging</h4>
                  <p>Applies detailed metadata to each content piece, enabling precise categorization and retrieval during evaluation.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Content Diversity</h4>
                  <p>Ensures representation across various Islamic topics including theology, jurisprudence, ethics, history, and contemporary issues.</p>
                </FeatureCard>
              </AgentFeatures>
            </AgentDescription>
          </AgentContainer>
          
          <AgentContainer>
            <AgentTitle variants={itemVariants}>Agent 1B: Question Generation</AgentTitle>
            <AgentDescription variants={itemVariants}>
              <p>
                Agent 1B specializes in creating diverse, challenging questions that effectively test language models' understanding of Islamic concepts. This agent works closely with Agent 1A to develop questions based on the curated content.
              </p>
              <p>Key responsibilities include:</p>
              <ul>
                <li>Generating questions across different difficulty levels and formats</li>
                <li>Creating questions that test factual knowledge, ethical reasoning, and contextual understanding</li>
                <li>Developing questions that can identify potential biases in model responses</li>
                <li>Ensuring questions are culturally sensitive and respectful</li>
              </ul>
              
              <AgentFeatures>
                <FeatureCard variants={itemVariants}>
                  <h4>Multi-format Questions</h4>
                  <p>Creates various question types including multiple-choice, open-ended, scenario-based, and comparative analysis questions.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Difficulty Scaling</h4>
                  <p>Implements a systematic approach to creating questions with graduated difficulty levels to thoroughly test model capabilities.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Bias Detection</h4>
                  <p>Designs specialized questions to identify potential biases or misrepresentations of Islamic concepts in model responses.</p>
                </FeatureCard>
              </AgentFeatures>
            </AgentDescription>
          </AgentContainer>
          
          <AgentContainer>
            <AgentTitle variants={itemVariants}>Agent 2: Evaluation Framework</AgentTitle>
            <AgentDescription variants={itemVariants}>
              <p>
                Agent 2 develops and implements the comprehensive evaluation framework that assesses language models across our four key metrics. This agent establishes the methodological foundation for our entire evaluation process.
              </p>
              <p>Key responsibilities include:</p>
              <ul>
                <li>Designing evaluation protocols for each of the four key metrics</li>
                <li>Establishing scoring criteria and benchmarks</li>
                <li>Implementing the lm-evaluation-harness integration</li>
                <li>Ensuring evaluation consistency across different models</li>
              </ul>
              
              <AgentFeatures>
                <FeatureCard variants={itemVariants}>
                  <h4>Multi-dimensional Assessment</h4>
                  <p>Evaluates models across knowledge accuracy, ethical understanding, bias detection, and source reliability dimensions.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Weighted Scoring</h4>
                  <p>Implements a sophisticated weighted scoring system that prioritizes critical aspects of Islamic knowledge representation.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Benchmark Calibration</h4>
                  <p>Establishes performance benchmarks through expert validation and continuous refinement of evaluation criteria.</p>
                </FeatureCard>
              </AgentFeatures>
            </AgentDescription>
          </AgentContainer>
          
          <AgentContainer>
            <AgentTitle variants={itemVariants}>Agent 3: Model Evaluation</AgentTitle>
            <AgentDescription variants={itemVariants}>
              <p>
                Agent 3 conducts the actual evaluation of language models using the framework established by Agent 2 and the content prepared by Agents 1A and 1B. This agent is responsible for generating the performance data that populates our leaderboard.
              </p>
              <p>Key responsibilities include:</p>
              <ul>
                <li>Running evaluation protocols on various language models</li>
                <li>Collecting and processing model responses</li>
                <li>Calculating scores across all evaluation metrics</li>
                <li>Identifying performance patterns and insights</li>
              </ul>
              
              <AgentFeatures>
                <FeatureCard variants={itemVariants}>
                  <h4>Automated Testing</h4>
                  <p>Implements automated testing pipelines to efficiently evaluate multiple models using standardized protocols.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Response Analysis</h4>
                  <p>Applies sophisticated analysis techniques to extract meaningful insights from model responses.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Performance Tracking</h4>
                  <p>Maintains detailed records of model performance over time to track improvements and identify trends.</p>
                </FeatureCard>
              </AgentFeatures>
            </AgentDescription>
          </AgentContainer>
          
          <AgentContainer>
            <AgentTitle variants={itemVariants}>Agent 4: Leaderboard Maintenance</AgentTitle>
            <AgentDescription variants={itemVariants}>
              <p>
                Agent 4 manages the leaderboard system that presents evaluation results in an accessible, informative format. This agent ensures that users can easily identify which models best align with Islamic principles and accurately represent Islamic knowledge.
              </p>
              <p>Key responsibilities include:</p>
              <ul>
                <li>Organizing and presenting evaluation results</li>
                <li>Implementing the grading system (A+ to F)</li>
                <li>Updating the leaderboard with new model evaluations</li>
                <li>Providing detailed performance breakdowns and comparisons</li>
              </ul>
              
              <AgentFeatures>
                <FeatureCard variants={itemVariants}>
                  <h4>Interactive Visualization</h4>
                  <p>Creates intuitive data visualizations that allow users to explore model performance across different metrics.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Comparative Analysis</h4>
                  <p>Enables side-by-side comparison of multiple models to highlight relative strengths and weaknesses.</p>
                </FeatureCard>
                <FeatureCard variants={itemVariants}>
                  <h4>Transparent Methodology</h4>
                  <p>Provides clear documentation of evaluation methodologies to ensure users understand how grades are determined.</p>
                </FeatureCard>
              </AgentFeatures>
            </AgentDescription>
          </AgentContainer>
        </motion.div>
        
        <NavigationButton 
          href="#appendix"
          whileHover={{ y: -5 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 0.7 }}
        >
          ↓
        </NavigationButton>
      </TechnicalContainer>
    </TechnicalSection>
  );
};

export default TechnicalComponents; 