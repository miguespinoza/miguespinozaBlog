---
title: React conceptual introduction
date: "2019-02-05T03:52:33.219Z"
---

In this article I want to explain the key parts of react, without building a react application, I think concepts are very important to understand why react is good and why it works that way.

## Architecture flexibility

React is not a framework, its just a UI library, this is the single best thing about react, it leaves everything else about the application to you and with great power comes great responsibility.

Things like from code folder structure, how to style your components, state handling, server requests, build tools, file extensions and more its up to you the programmer to decide, this allows for very powerful design patterns to be used, and even to use more than one in a single application.

This means that you will have to make hard decisions, there are community driven tools that offer those decisions already taken,

  - [Create-react-app](https://facebook.github.io/create-react-app/) it’s the biggest one, very powerful, easy to setup, very well documented, however it does not cover everything, its unopinionated about things like async requests, state management, and folder structure.

  - [React boilerplate](https://www.reactboilerplate.com/) it’s by far the most feature packed tool out there, I respect all their design choices. But it comes with a very steep learning curve, I think it’s only sorted for big structured, transactional sites.

This flexibility is bounded as explained by Dan Abramov [here](https://overreacted.io/react-as-a-ui-runtime/) by two things:

  - Stability: a site cannot re structure every element at every moment, all the time

  - Regularity: every UI element must behave consistently

React will do it's best if you really want to do one of the above but, it won’t be good.

Having said that, the things that you can build with react are surely big, powerful UI patterns to engage your users and deliver good value.

## React Elements

Are the building blocks of react applications, an element describes what you see on the screen.

As a programmer you normally don’t interact directly with them, there are several react mechanisms that create and delete them as needed. (Yes, only create and delete because they are immutable).

## React Components

This approach allows you to split the UI into independent pieces that can be composed together to form a web experience.

The easiest way to think of a component is like a LEGO brick, they accept props (short for properties) and return react elements.


## Tips

- The component name is very important, make it self explanatory.
- Make your components as atomic as you can.
- Keep the domain knowledge outside the component class (or function).
	-  Let’s say you are building a web store and have a list of products; each product has to be parsed and you have to color the cell depending on some business logic. You could build a function in the component and called it from the render method, but that its going to cost you in the future.
	-  Keep that logic in a separate file and import it. With this approach both things are simpler and also can be tested easily.

> Written with [StackEdit](https://stackedit.io/).

