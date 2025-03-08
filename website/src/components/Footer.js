import React from 'react';
import styled from 'styled-components';

const FooterSection = styled.footer`
  background-color: var(--primary-color);
  color: #fff;
  padding: 4rem 2rem 2rem;
`;

const FooterContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 3rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    text-align: center;
  }
`;

const FooterColumn = styled.div`
  h3 {
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    color: var(--secondary-color);
  }
  
  p {
    margin-bottom: 1rem;
    opacity: 0.8;
    line-height: 1.6;
  }
  
  ul {
    list-style: none;
    padding: 0;
  }
  
  li {
    margin-bottom: 0.8rem;
  }
  
  a {
    color: #fff;
    opacity: 0.8;
    transition: opacity 0.3s ease;
    
    &:hover {
      opacity: 1;
      text-decoration: underline;
    }
  }
`;

const Copyright = styled.div`
  text-align: center;
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  opacity: 0.7;
  font-size: 0.9rem;
`;

const Footer = () => {
  return (
    <FooterSection>
      <FooterContainer>
        <FooterColumn>
          <h3>Islamic GenAI Guild LLM Evaluation Project</h3>
          <p>
            A project dedicated to evaluating and enhancing AI language models through Islamic knowledge and ethical guidelines.
          </p>
          <p>
            Our mission is to ensure AI tools align with Islamic principles and accurately represent Islamic knowledge.
          </p>
        </FooterColumn>
        
        <FooterColumn>
          <h3>Quick Links</h3>
          <ul>
            <li><a href="#about">About the Project</a></li>
            <li><a href="#objectives">Our Objectives</a></li>
            <li><a href="#models">Evaluated Models</a></li>
            <li><a href="#evaluation">Evaluation Results</a></li>
            <li><a href="https://github.com/zainkhatri/mcc-genai-guild" target="_blank" rel="noopener noreferrer">GitHub Repository</a></li>
          </ul>
        </FooterColumn>
        
        <FooterColumn>
          <h3>Contact</h3>
          <p>
            Have questions or want to contribute to the project? Reach out to us through our GitHub repository or contact us directly.
          </p>
          <p>
            <a href="mailto:contactus@mccsandiego.org">contactus@mccsandiego.org</a>
          </p>
        </FooterColumn>
      </FooterContainer>
      
      <Copyright>
        &copy; 2025 Islamic GenAI Guild LLM Evaluation Project. All rights reserved.
      </Copyright>
    </FooterSection>
  );
};

export default Footer; 