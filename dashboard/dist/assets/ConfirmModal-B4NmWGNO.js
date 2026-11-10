import{e as p,u as h,j as e,h as u}from"./index-mJJcthPd.js";/**
 * @license lucide-react v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const x=p("TriangleAlert",[["path",{d:"m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3",key:"wmoenq"}],["path",{d:"M12 9v4",key:"juzpu7"}],["path",{d:"M12 17h.01",key:"p32p05"}]]);function y({open:o,title:c,message:t,onConfirm:l,onCancel:s,confirmText:i,cancelText:r,confirmDisabled:n}){const{language:d}=h(),a=u(d);return o?e.jsx("div",{className:"modal-overlay",onClick:n?void 0:s,children:e.jsxs("div",{className:"modal-box confirm-modal",onClick:m=>m.stopPropagation(),children:[e.jsx("div",{className:"confirm-icon",children:e.jsx(x,{size:40})}),e.jsx("h4",{children:c}),e.jsx("p",{style:{whiteSpace:"pre-wrap"},children:t}),e.jsxs("div",{className:"confirm-actions",children:[e.jsx("button",{type:"button",className:"btn-secondary",onClick:s,disabled:n,children:r||a("common.cancel")}),e.jsx("button",{type:"button",className:"btn-danger",onClick:l,disabled:n,children:i||a("common.confirmDelete")})]})]})}):null}export{y as C,x as T};
