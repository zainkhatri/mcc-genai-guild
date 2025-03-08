import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import Hero from './components/Hero';
import About from './components/About';
import Objectives from './components/Objectives';
import Models from './components/Models';
import Evaluation from './components/Evaluation';
import TechnicalComponents from './components/TechnicalComponents';
import Ethics from './components/Ethics';
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
      <Models />
      <Evaluation />
      <TechnicalComponents />
      <Ethics />
      <Appendix />
      <Footer />
    </AppContainer>
  );
}

export default App;