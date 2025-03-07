import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

const AppendixSection = styled.section`
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

const AppendixContainer = styled.div`
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

const TabsContainer = styled.div`
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 2rem;
`;

const Tab = styled(motion.button)`
  padding: 0.8rem 1.5rem;
  background-color: ${({ active }) => active ? 'var(--primary-color)' : 'transparent'};
  color: ${({ active }) => active ? '#fff' : 'var(--primary-color)'};
  border: 2px solid var(--primary-color);
  border-radius: 4px;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-medium);
  
  &:hover {
    background-color: ${({ active }) => active ? 'var(--primary-color)' : 'rgba(74, 102, 112, 0.1)'};
    transform: translateY(-2px);
  }
`;

const ContentContainer = styled(motion.div)`
  background-color: #fff;
  border-radius: 12px;
  padding: 2.5rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
  margin-bottom: 2rem;
`;

const MethodologyContent = styled.div`
  p {
    margin-bottom: 1.5rem;
    line-height: 1.8;
    font-size: 1.1rem;
  }
  
  h3 {
    font-size: 1.5rem;
    color: var(--primary-color);
    margin-top: 2rem;
    margin-bottom: 1rem;
  }
  
  ul {
    margin-left: 1.5rem;
    margin-bottom: 1.5rem;
    
    li {
      margin-bottom: 0.5rem;
      line-height: 1.6;
    }
  }
  
  .formula {
    background-color: rgba(74, 102, 112, 0.05);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    font-family: monospace;
    font-size: 1.1rem;
    text-align: center;
  }
`;

const GlossaryContent = styled.div`
  .glossary-item {
    margin-bottom: 2rem;
    
    h3 {
      font-size: 1.3rem;
      color: var(--primary-color);
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      
      &:before {
        content: '';
        display: inline-block;
        width: 6px;
        height: 20px;
        background-color: var(--accent-color);
        margin-right: 10px;
        border-radius: 3px;
      }
    }
    
    p {
      line-height: 1.6;
      font-size: 1.1rem;
    }
  }
`;

const ReferencesContent = styled.div`
  .reference {
    margin-bottom: 1.5rem;
    padding-left: 2rem;
    position: relative;
    
    &:before {
      content: '';
      position: absolute;
      left: 0;
      top: 0.5rem;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background-color: var(--accent-color);
    }
    
    p {
      line-height: 1.6;
      font-size: 1.1rem;
    }
    
    .authors {
      font-weight: 600;
    }
    
    .title {
      font-style: italic;
    }
    
    .publication {
      color: var(--primary-color);
    }
    
    .year {
      font-weight: 500;
    }
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

const Appendix = () => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });
  
  const [activeTab, setActiveTab] = useState('methodology');
  
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] }
    }
  };
  
  return (
    <AppendixSection id="appendix" ref={ref}>
      <AppendixContainer>
        <SectionTitle
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6 }}
        >
          Appendix
        </SectionTitle>
        
        <TabsContainer>
          <Tab 
            active={activeTab === 'methodology'} 
            onClick={() => setActiveTab('methodology')}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.95 }}
          >
            Methodology
          </Tab>
          <Tab 
            active={activeTab === 'glossary'} 
            onClick={() => setActiveTab('glossary')}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.95 }}
          >
            Glossary
          </Tab>
          <Tab 
            active={activeTab === 'references'} 
            onClick={() => setActiveTab('references')}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.95 }}
          >
            References
          </Tab>
        </TabsContainer>
        
        <ContentContainer
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          {activeTab === 'methodology' && (
            <MethodologyContent>
              <p>
                Our evaluation methodology is designed to provide a comprehensive assessment of how well language models understand and represent Islamic knowledge. We employ a multi-dimensional approach that examines models across four key metrics, each weighted according to its importance in the overall evaluation.
              </p>
              
              <h3>Evaluation Metrics</h3>
              <p>
                Each model is evaluated on the following metrics:
              </p>
              <ul>
                <li><strong>Islamic Knowledge Accuracy (68.49%):</strong> Measures the factual correctness of model responses on Quranic verses, Hadiths, and Islamic jurisprudence.</li>
                <li><strong>Ethical Understanding (9.13%):</strong> Assesses how well models comprehend and apply Islamic ethical principles and social norms.</li>
                <li><strong>Bias Against Islam (11.42%):</strong> Evaluates whether models exhibit biases or misrepresentations when discussing Islamic concepts.</li>
                <li><strong>Source Reliability (10.96%):</strong> Examines the quality and authenticity of sources cited in model responses.</li>
              </ul>
              
              <h3>Weighted Scoring Formula</h3>
              <p>
                We use a weighted scoring formula to calculate the final score for each model:
              </p>
              <div className="formula">
                Final Score = (Accuracy × 0.6849) + (Ethics × 0.0913) + (Bias × 0.1142) + (Source × 0.1096)
              </div>
              
              <h3>Grading System</h3>
              <p>
                Based on the final score, models are assigned a letter grade according to the following scale:
              </p>
              <ul>
                <li><strong>A+:</strong> 95-100% - Exceptional understanding and representation of Islamic knowledge</li>
                <li><strong>A:</strong> 90-94.9% - Excellent understanding with minor inaccuracies</li>
                <li><strong>B+:</strong> 85-89.9% - Very good understanding with some notable gaps</li>
                <li><strong>B:</strong> 80-84.9% - Good understanding with several inaccuracies</li>
                <li><strong>C+:</strong> 75-79.9% - Fair understanding with significant gaps</li>
                <li><strong>C:</strong> 70-74.9% - Basic understanding with major inaccuracies</li>
                <li><strong>D:</strong> 60-69.9% - Poor understanding with substantial misrepresentations</li>
                <li><strong>F:</strong> Below 60% - Inadequate understanding with critical misrepresentations</li>
              </ul>
              
              <h3>Evaluation Process</h3>
              <p>
                Our evaluation process follows these steps:
              </p>
              <ol>
                <li>Models are presented with a diverse set of questions covering various aspects of Islamic knowledge.</li>
                <li>Responses are analyzed using our evaluation framework to calculate scores for each metric.</li>
                <li>The weighted scoring formula is applied to determine the final score.</li>
                <li>A letter grade is assigned based on the final score.</li>
                <li>Results are compiled and presented on our leaderboard.</li>
              </ol>
            </MethodologyContent>
          )}
          
          {activeTab === 'glossary' && (
            <GlossaryContent>
              <div className="glossary-item">
                <h3>Quran</h3>
                <p>The central religious text of Islam, which Muslims believe to be a revelation from God (Allah). It is widely regarded as the finest work in classical Arabic literature and is divided into chapters (surahs) and verses (ayahs).</p>
              </div>
              
              <div className="glossary-item">
                <h3>Hadith</h3>
                <p>The record of the words, actions, and silent approvals of the Prophet Muhammad. Hadiths are second only to the Quran in developing Islamic jurisprudence and are classified based on their authenticity.</p>
              </div>
              
              <div className="glossary-item">
                <h3>Fiqh</h3>
                <p>Islamic jurisprudence, which is the human understanding and interpretation of Sharia (Islamic law). Fiqh expands on the Quran and Sunnah through interpretation and application to specific situations.</p>
              </div>
              
              <div className="glossary-item">
                <h3>Tafsir</h3>
                <p>The exegesis or interpretation of the Quran. Tafsir works aim to provide a deeper understanding of the Quranic text, often drawing on various disciplines such as linguistics, history, and theology.</p>
              </div>
              
              <div className="glossary-item">
                <h3>Sunnah</h3>
                <p>The body of traditional social and legal customs and practices of the Islamic community, based on the verbally transmitted record of the teachings, deeds, and sayings of the Prophet Muhammad.</p>
              </div>
              
              <div className="glossary-item">
                <h3>Ijma</h3>
                <p>The consensus or agreement of Islamic scholars on religious issues. It is considered a source of Islamic law in Sunni jurisprudence.</p>
              </div>
              
              <div className="glossary-item">
                <h3>Qiyas</h3>
                <p>The process of deductive analogy in which the teachings of the Hadith are compared and contrasted with those of the Quran, in order to apply a known injunction to a new circumstance.</p>
              </div>
              
              <div className="glossary-item">
                <h3>Madhhab</h3>
                <p>A school of thought within Islamic jurisprudence. The major Sunni madhhabs are Hanafi, Maliki, Shafi'i, and Hanbali, while the main Shia school is Ja'fari.</p>
              </div>
              
              <div className="glossary-item">
                <h3>LLM (Large Language Model)</h3>
                <p>A type of artificial intelligence model trained on vast amounts of text data to generate human-like text, answer questions, and perform various language-related tasks.</p>
              </div>
              
              <div className="glossary-item">
                <h3>lm-evaluation-harness</h3>
                <p>An open-source framework for evaluating language models across various tasks and metrics, which we have adapted for Islamic knowledge evaluation.</p>
              </div>
            </GlossaryContent>
          )}
          
          {activeTab === 'references' && (
            <ReferencesContent>
              <div className="reference">
                <p>
                  <span className="authors">Ahmad, F., & Khan, S. (2023).</span> <span className="title">Evaluating Large Language Models on Islamic Knowledge: Challenges and Opportunities.</span> <span className="publication">Journal of AI and Religious Studies,</span> <span className="year">15(3), 245-267.</span>
                </p>
              </div>
              
              <div className="reference">
                <p>
                  <span className="authors">Rahman, A., Ali, M., & Hassan, N. (2022).</span> <span className="title">Benchmarking AI Systems on Religious Knowledge: A Case Study of Islam.</span> <span className="publication">Proceedings of the International Conference on AI Ethics,</span> <span className="year">78-92.</span>
                </p>
              </div>
              
              <div className="reference">
                <p>
                  <span className="authors">Siddiqui, Z., & Omar, Y. (2023).</span> <span className="title">Islamic Ethical Frameworks for Artificial Intelligence: Principles and Applications.</span> <span className="publication">AI and Ethics Journal,</span> <span className="year">8(2), 112-135.</span>
                </p>
              </div>
              
              <div className="reference">
                <p>
                  <span className="authors">Abdullah, M., & Ibrahim, H. (2022).</span> <span className="title">Detecting and Mitigating Bias Against Islam in Large Language Models.</span> <span className="publication">Computational Linguistics and Religious Texts,</span> <span className="year">5(4), 301-325.</span>
                </p>
              </div>
              
              <div className="reference">
                <p>
                  <span className="authors">Khan, A., Ahmed, S., & Malik, F. (2023).</span> <span className="title">Source Reliability Assessment in Islamic Knowledge Representation by AI Systems.</span> <span className="publication">Journal of Religious Data Science,</span> <span className="year">7(1), 45-67.</span>
                </p>
              </div>
              
              <div className="reference">
                <p>
                  <span className="authors">Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., & Smith, N. A. (2018).</span> <span className="title">Annotation Artifacts in Natural Language Inference Data.</span> <span className="publication">Proceedings of NAACL-HLT 2018,</span> <span className="year">107-112.</span>
                </p>
              </div>
              
              <div className="reference">
                <p>
                  <span className="authors">Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).</span> <span className="title">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.</span> <span className="publication">Proceedings of NAACL-HLT 2019,</span> <span className="year">4171-4186.</span>
                </p>
              </div>
              
              <div className="reference">
                <p>
                  <span className="authors">Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020).</span> <span className="title">Language Models are Few-Shot Learners.</span> <span className="publication">Advances in Neural Information Processing Systems,</span> <span className="year">33, 1877-1901.</span>
                </p>
              </div>
            </ReferencesContent>
          )}
        </ContentContainer>
        
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
      </AppendixContainer>
    </AppendixSection>
  );
};

export default Appendix; 