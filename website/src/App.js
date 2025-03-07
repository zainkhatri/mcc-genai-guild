import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import Hero from './components/Hero';
import About from './components/About';
import Objectives from './components/Objectives';
import Evaluation from './components/Evaluation';
import Models from './components/Models';
import TechnicalComponents from './components/TechnicalComponents';
import Appendix from './components/Appendix';
import Footer from './components/Footer';

const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;

function App() {
  return (
    <AppContainer>
      <Hero />
      <About />
      <Objectives />
      <Evaluation />
      <Models />
      <TechnicalComponents />
      <Appendix />
      <Footer />
    </AppContainer>
  );
}

export default App;