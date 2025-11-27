export const buildServiceError = async (service: string, response: Response): Promise<Error> => {
  const snippet = await response
    .text()
    .then((text) => text.slice(0, 200))
    .catch(() => "");

  const details = [`HTTP ${response.status}`];
  if (response.statusText) {
    details.push(response.statusText);
  }
  if (snippet) {
    details.push(`Body: ${snippet}`);
  }

  return new Error(`${service} failed: ${details.join(" | ")}`);
};
