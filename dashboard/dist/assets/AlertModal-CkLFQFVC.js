import{e as i,r as l,j as a}from"./index-DF5X5cCi.js";import{I as d}from"./info-Be7rb6g8.js";/**
 * @license lucide-react v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const h=i("BookOpen",[["path",{d:"M12 7v14",key:"1akyts"}],["path",{d:"M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z",key:"ruj8y"}]]);function u({open:r,title:s,message:t,okText:n="OK",onClose:o,danger:e}){return r?l.createPortal(a.jsx("div",{className:"modal-overlay",onClick:o,children:a.jsxs("div",{className:"modal-box confirm-modal",onClick:c=>c.stopPropagation(),children:[a.jsx("div",{className:"confirm-icon",style:e?{color:"var(--danger)"}:void 0,children:a.jsx(d,{size:40})}),a.jsx("h4",{children:s}),a.jsx("p",{style:{whiteSpace:"pre-wrap"},children:t}),a.jsx("div",{className:"confirm-actions",children:a.jsx("button",{type:"button",className:e?"btn-danger":"btn-primary",onClick:o,autoFocus:!0,children:n})})]})}),document.body):null}export{u as A,h as B};
