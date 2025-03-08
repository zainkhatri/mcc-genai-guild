import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

const EthicsSection = styled.section`
  padding: 6rem 2rem;
  background: linear-gradient(135deg, var(--cream) 0%, var(--cream-light) 100%);
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

const EthicsContainer = styled.div`
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

const EthicsContent = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  
  @media (min-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const EthicsCard = styled(motion.div)`
  background: white;
  border-radius: 10px;
  padding: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease;
  
  h3 {
    color: var(--primary-color);
    margin-bottom: 1rem;
    font-size: 1.5rem;
  }
  
  p {
    color: var(--text-color);
    line-height: 1.6;
    margin-bottom: 1rem;
  }
  
  ul {
    padding-left: 1.5rem;
    margin-bottom: 1rem;
  }
  
  li {
    margin-bottom: 0.5rem;
    line-height: 1.6;
  }
`;

const Ethics = () => {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <EthicsSection id="ethics">
      <EthicsContainer ref={ref}>
        <SectionTitle
          initial={{ opacity: 0, y: -20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
        >
          Ethics in AI
        </SectionTitle>
        
        <EthicsContent
          as={motion.div}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          <EthicsCard variants={itemVariants}>
            <p>
              At the Islamic GenAI Guild, ethics is central to our mission. We aim to ensure that AI tools align with Islamic principles and contribute positively to society. Here are the key considerations guiding our work:
            </p>
            
            <h3>Accuracy and Integrity</h3>
            <p>
              AI systems should provide accurate, reliable, and verifiable information, particularly when addressing sensitive topics like Fiqh or Hadith.
            </p>
            
            <h3>Avoiding Bias</h3>
            <p>
              AI tools must be free from cultural, religious, or political biases that could misrepresent Islamic teachings.
            </p>
            
            <h3>Privacy and Security</h3>
            <p>
              User data must be protected, and AI systems should not exploit or misuse personal information.
            </p>
            
            <h3>Ethical Limitations</h3>
            <p>
              AI should not be used to issue fatwas or make religious decisions autonomously.
              Tools must clearly indicate their limitations and the need for human oversight.
            </p>
            
            <h3>Transparency</h3>
            <p>
              All AI models and datasets should be developed with transparency, ensuring their processes and sources are clear.
            </p>
            
            <h3>Accountability</h3>
            <p>
              Developers and users of AI tools must be accountable for their outputs, ensuring they align with ethical and Islamic standards.
            </p>
            
            <p>
              By emphasizing these principles, we strive to create AI systems that respect and uphold the values of Islam while serving as a valuable resource for education and research.
            </p>
          </EthicsCard>
        </EthicsContent>
      </EthicsContainer>
    </EthicsSection>
  );
};

export default Ethics; 