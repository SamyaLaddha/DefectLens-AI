import React from 'react';
import { Link } from 'react-router-dom';
import HeroSlider from '../components/HeroSlider';

const CATEGORIES = [
  { name: 'Bottle', looks: 'Top-down view of a bottle opening: rim damage, contamination.', note: 'Verdict and heatmap are both usable.' },
  { name: 'Cable', looks: 'Cable end face: missing or swapped wires, cuts, damaged insulation.', note: 'Trust the heatmap more than the pass/fail verdict.' },
  { name: 'Screw', looks: 'Single screw on a plain background: scratches, thread and head damage.', note: 'Trust the heatmap more than the pass/fail verdict.' },
];

function SectionHead({ num, kicker, title, lede }) {
  return (
    <div className="section__head">
      <div className="section__num" aria-hidden="true">{num}</div>
      <div>
        <span className="kicker">{kicker}</span>
        <h2 className="display section__title">{title}</h2>
        {lede && <p className="section__lede">{lede}</p>}
      </div>
    </div>
  );
}

function Landing() {
  return (
    <>
      {/* HERO */}
      <section className="hero">
        <div className="wrap hero__grid">
          <div>
            <span className="kicker">Industrial visual inspection</span>
            <h1 className="display">
              Find the flaw.<br />
              <span className="mark"><em>See where.</em></span>
            </h1>
            <p className="hero__lede">
              Drop in a photo of a part. DefectLens says whether something is
              off and highlights exactly where, having only ever seen good parts.
            </p>
            <div className="hero__cta">
              <Link to="/try" className="btn">Inspect a part →</Link>
              <a href="#how" className="btn btn--ghost">How it works</a>
            </div>
            <p className="kicker hero__fine">No sign-up · No install · Runs in your browser</p>
          </div>
          <HeroSlider />
        </div>
      </section>

      <div className="strip">
        <div className="wrap strip__row">
          <span>Trained on normal images only</span>
          <span>No labeled defects required</span>
          <span>Verdict + score + heatmap</span>
          <span>Single image or batch of 4–12</span>
        </div>
      </div>

      {/* 01 PROBLEM */}
      <section className="section">
        <div className="wrap">
          <SectionHead
            num="01"
            kicker="The problem"
            title={<>Defects are rare. <em>Examples</em> are rarer.</>}
            lede="A factory makes thousands of good parts for every bad one, and each bad one looks different. Collecting and labeling enough defect photos to train a normal classifier is slow and never complete. So this model never sees one."
          />
        </div>
      </section>

      {/* 02 HOW */}
      <section className="section section--alt" id="how">
        <div className="wrap">
          <SectionHead
            num="02"
            kicker="How it works"
            title={<>A student that only learned <em>normal</em>.</>}
            lede="DefectLens uses Student-Teacher Feature Pyramid Matching (STFPM). Disagreement between two networks is the signal."
          />
          <div className="steps">
            <div className="step">
              <span className="step__n">STEP 1</span>
              <h3>A teacher that knows images</h3>
              <p>A frozen, pretrained network turns every image into multi-scale features. It never changes.</p>
            </div>
            <div className="step">
              <span className="step__n">STEP 2</span>
              <h3>A student learns “good”</h3>
              <p>A second network is trained to copy the teacher, but only on photos of defect-free parts.</p>
            </div>
            <div className="step">
              <span className="step__n">STEP 3</span>
              <h3>Disagreement is the defect</h3>
              <p>On a flawed part the student can’t match the teacher. Where they differ becomes the heatmap, and its peak becomes the score.</p>
            </div>
          </div>
        </div>
      </section>

      {/* 03 OUTPUTS */}
      <section className="section">
        <div className="wrap">
          <SectionHead
            num="03"
            kicker="What you get"
            title={<>Not just a <em>yes or no</em>.</>}
          />
          <div className="outputs">
            <div className="card">
              <span className="kicker">Output A</span>
              <h3>Verdict</h3>
              <p>Normal or anomalous, decided by comparing the score to a threshold set on held-out normal images.</p>
            </div>
            <div className="card">
              <span className="kicker">Output B</span>
              <h3>Anomaly score</h3>
              <p>The raw number behind the verdict, shown next to the threshold, so you can see how close the call was.</p>
            </div>
            <div className="card">
              <span className="kicker">Output C</span>
              <h3>Defect heatmap</h3>
              <p>An overlay showing where the model sees a mismatch, so a person can check it in seconds.</p>
            </div>
            <div className="card">
              <span className="kicker">Output D</span>
              <h3>Batch inspection</h3>
              <p>Submit 4 to 12 images at once and get a pass/fail summary with a result for each image.</p>
            </div>
          </div>
        </div>
      </section>

      {/* 04 CAPABILITIES */}
      <section className="section section--alt" id="capabilities">
        <div className="wrap">
          <SectionHead
            num="04"
            kicker="Current capabilities"
            title={<>Three categories. <em>Honestly</em> scoped.</>}
            lede="Each category has its own trained model."
          />
          <table className="cap">
            <thead>
              <tr><th>Category</th><th>What it inspects</th><th>Status</th><th>Read it like this</th></tr>
            </thead>
            <tbody>
              {CATEGORIES.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>{c.looks}</td>
                  <td><span className="pill">LIVE</span></td>
                  <td>{c.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            <strong>Scope</strong><br />
            Models are trained and evaluated on the public MVTec AD benchmark. They have not been
            validated on a real production line, and photos that differ from the benchmark's framing
            and lighting may give unreliable results. This is a research demo, not a certified inspection system.
          </p>
        </div>
      </section>

      {/* 05 CTA */}
      <section className="final">
        <div className="wrap">
          <span className="kicker" style={{ color: 'var(--color-ink)' }}>05 · Your turn</span>
          <h2 className="display" style={{ marginTop: 14 }}>Try it on a <em>part</em>.</h2>
          <Link to="/try" className="btn">Open the inspection bench →</Link>
          <p className="kicker" style={{ color: 'var(--color-ink)', marginTop: 20 }}>Free · No account · Nothing to install</p>
        </div>
      </section>
    </>
  );
}

export default Landing;
